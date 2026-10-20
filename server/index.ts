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
import { ModelStudio } from './studio.ts';
import { NewLife } from './new-life.ts';
import { NEW_LIFE_PRESETS, type NewLifeSize } from '../shared/api.ts';
import { tensorBackend } from './numerics.ts';

export type { Request, Response } from './http.ts';

const projectDirectory = fileURLToPath(new URL('../', import.meta.url));
const buildDirectory = path.join(projectDirectory, 'dist');
// Vite forwards browser Origin unchanged, even if the proxy rewrites Host.
const developmentOrigins =
  process.env.NODE_ENV === 'development' ? ['http://localhost:5173', 'http://127.0.0.1:5173'] : [];
const studio = new ModelStudio(path.join(projectDirectory, 'data/checkpoint.json'));
const lives = new Map<NewLifeSize, NewLife>();
function getLife(size: unknown = 'small'): NewLife {
  if (typeof size !== 'string' || !Object.hasOwn(NEW_LIFE_PRESETS, size)) {
    throw new Error('Invalid New life model size');
  }
  const selected = size as NewLifeSize;
  let life = lives.get(selected);
  if (!life) {
    const suffix = selected === 'small' ? '' : `-${selected}`;
    life = new NewLife(
      path.join(projectDirectory, `data/new-life-checkpoint${suffix}.json`),
      selected,
    );
    lives.set(selected, life);
  }
  return life;
}

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
    sendJson(response, 200, await getLife().getStatus());
    return;
  }

  if (request.method === 'GET' && pathname === '/api/status') {
    const checkpoint = await studio.getCheckpointInfo();
    sendJson(response, 200, { ...studio.getStatus(), checkpoint });
    return;
  }

  if (request.method !== 'POST') {
    sendJson(response, 404, { error: 'Unknown endpoint' });
    return;
  }

  const body = await readJsonBody(request);

  switch (pathname) {
    case '/api/new-life/settings':
      sendJson(response, 200, await getLife(body.size).getStatus());
      return;
    case '/api/new-life/message':
      sendJson(response, 200, await getLife(body.size).message(body));
      return;
    case '/api/new-life/teach':
      sendJson(response, 200, await getLife(body.size).teach(body));
      return;
    case '/api/train':
      studio.startTraining(body);
      sendJson(response, 202, studio.getStatus());
      return;

    case '/api/stop':
      studio.stopTraining();
      sendJson(response, 200, studio.getStatus());
      return;

    case '/api/generate':
      sendJson(response, 200, studio.generate(body));
      return;

    case '/api/chat':
      sendJson(response, 200, studio.chat(body));
      return;

    case '/api/checkpoint/save':
      await studio.saveCheckpoint();
      sendJson(response, 200, { message: 'Checkpoint saved to data/checkpoint.json' });
      return;

    case '/api/checkpoint/delete':
      await studio.deleteCheckpoint();
      sendJson(response, 200, {
        message: 'Saved checkpoint deleted. The loaded model is still available.',
      });
      return;

    case '/api/checkpoint/load':
      await studio.loadCheckpoint();
      sendJson(response, 200, studio.getStatus());
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
    console.log(`Tensor backend: ${tensorBackend} (CPU, Float64)`);
  });
}
