import type { IncomingMessage } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { getErrorMessage } from './validation.ts';
import type { JsonObject } from './validation.ts';

// Structural interfaces let tests exercise requests without opening network sockets.
export type Request = AsyncIterable<Uint8Array> &
  Pick<IncomingMessage, 'url' | 'method' | 'headers'>;

export interface Response {
  writeHead(status: number, headers: Record<string, string>): unknown;
  end(data: string | Uint8Array): unknown;
}

const MAX_REQUEST_BYTES = 21_000_000;
const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.svg': 'image/svg+xml',
};

export function sendJson(response: Response, status: number, data: unknown): void {
  response.writeHead(status, {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-store',
  });
  response.end(JSON.stringify(data));
}

export function isCrossOriginRequest(
  request: Request,
  allowedOrigins: readonly string[] = [],
): boolean {
  const { origin, host } = request.headers;
  return Boolean(origin && origin !== `http://${host}` && !allowedOrigins.includes(origin));
}

export async function readJsonBody(request: Request): Promise<JsonObject> {
  let receivedBytes = 0;
  const chunks: Uint8Array[] = [];

  for await (const chunk of request) {
    receivedBytes += chunk.length;

    if (receivedBytes > MAX_REQUEST_BYTES) {
      throw new Error('Request exceeds 21 MB');
    }

    chunks.push(chunk);
  }

  const contents = Buffer.concat(chunks).toString();
  const body: unknown = JSON.parse(contents || '{}');

  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    throw new Error('Request body must be a JSON object');
  }

  return body as JsonObject;
}

export async function serveFrontend(
  response: Response,
  pathname: string,
  buildDirectory: string,
): Promise<void> {
  const relativePath = decodeURIComponent(pathname).replace(/^\/+/, '');
  let filePath = path.resolve(buildDirectory, relativePath || 'index.html');

  if (!filePath.startsWith(buildDirectory + path.sep)) {
    sendJson(response, 404, { error: 'Not found' });
    return;
  }

  let contents: Buffer;

  try {
    contents = await readFile(filePath);
  } catch {
    // Client-side routes use the same HTML entry point as the home page.
    filePath = path.join(buildDirectory, 'index.html');
    contents = await readFile(filePath);
  }

  const contentType = CONTENT_TYPES[path.extname(filePath)] ?? 'application/octet-stream';
  response.writeHead(200, { 'Content-Type': contentType });
  response.end(contents);
}

export function sendRequestError(response: Response, error: unknown): void {
  const fileNotFound = error instanceof Error && 'code' in error && error.code === 'ENOENT';
  const message = fileNotFound
    ? 'No checkpoint or production build found. Save a checkpoint or run npm run build.'
    : getErrorMessage(error);

  sendJson(response, 400, { error: message });
}
