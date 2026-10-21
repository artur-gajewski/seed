import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  isCrossOriginRequest,
  readJsonBody,
  sendJson,
  sendRequestError,
  serveFrontend,
} from './http.ts';
import type { Request, Response } from './http.ts';
import { TorchStudio } from './torch-studio.ts';

export type { Request, Response } from './http.ts';

const projectDirectory = fileURLToPath(new URL('../', import.meta.url));
const buildDirectory = path.join(projectDirectory, 'dist');
// Vite forwards browser Origin unchanged, even if the proxy rewrites Host.
const developmentOrigins =
  process.env.NODE_ENV === 'development' ? ['http://localhost:5173', 'http://127.0.0.1:5173'] : [];
export const studio = new TorchStudio(
  process.env.SEED_CHECKPOINT || path.join(projectDirectory, 'data/models/checkpoint.pt'),
);

async function handleApiRequest(
  request: Request,
  response: Response,
  pathname: string,
): Promise<void> {
  // This is a local tool: other websites must not be able to mutate its model.
  if (isCrossOriginRequest(request, developmentOrigins)) {
    sendJson(response, 403, { error: 'Cross-origin requests are disabled' });
    return;
  }

  if (request.method === 'GET' && pathname === '/api/new-life/status') {
    sendJson(response, 200, await studio.life('status'));
    return;
  }

  if (request.method === 'GET' && pathname === '/api/status') {
    sendJson(response, 200, await studio.getStatus());
    return;
  }

  if (request.method !== 'POST') {
    sendJson(response, 404, { error: 'Unknown endpoint' });
    return;
  }

  const body = await readJsonBody(request);

  switch (pathname) {
    case '/api/new-life/settings':
      sendJson(response, 200, await studio.life('status', body));
      return;
    case '/api/new-life/message':
      sendJson(response, 200, await studio.life('message', body));
      return;
    case '/api/new-life/teach':
      sendJson(response, 200, await studio.life('teach', body));
      return;
    case '/api/train':
      sendJson(response, 202, await studio.startTraining(body));
      return;

    case '/api/stop':
      sendJson(response, 200, await studio.stopTraining());
      return;

    case '/api/generate':
      sendJson(response, 200, await studio.generate(body));
      return;

    case '/api/chat':
      sendJson(response, 200, await studio.chat(body));
      return;

    case '/api/checkpoint/save':
      await studio.saveCheckpoint();
      sendJson(response, 200, {
        message: `Checkpoint saved to ${(await studio.getStatus()).checkpointFileName}`,
      });
      return;

    case '/api/checkpoint/delete':
      await studio.deleteCheckpoint();
      sendJson(response, 200, {
        message: 'Saved checkpoint deleted. The loaded model is still available.',
      });
      return;

    case '/api/checkpoint/load':
      sendJson(response, 200, await studio.loadCheckpoint(body));
      return;

    default:
      sendJson(response, 404, { error: 'Unknown endpoint' });
  }
}

export async function handler(request: Request, response: Response): Promise<void> {
  const url = new URL(request.url ?? '/', 'http://localhost');

  try {
    if (url.pathname.startsWith('/api/')) {
      await handleApiRequest(request, response, url.pathname);
      return;
    }

    if (request.method !== 'GET') {
      sendJson(response, 405, { error: 'Method not allowed' });
      return;
    }

    await serveFrontend(response, url.pathname, buildDirectory);
  } catch (error) {
    sendRequestError(response, error);
  }
}

export const server = http.createServer(handler);

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 3001);
  const host = process.env.HOST || '127.0.0.1';
  server.listen(port, host, () => {
    console.log(`Seed API running at http://${host}:${port}`);
    void studio
      .getStatus()
      .then((status) => {
        console.log(`Training backend: PyTorch (${status.backend?.device}, Float32)`);
      })
      .catch((error) => console.error(error.message));
  });
}

let shuttingDown = false;
async function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  server.close();
  try {
    await studio.close();
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
}
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  process.once('SIGINT', () => {
    void shutdown();
  });
  process.once('SIGTERM', () => {
    void shutdown();
  });
}
