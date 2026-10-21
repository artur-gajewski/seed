import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import type { StudioStatus, GenerateResponse, ChatResponse } from '../shared/api.ts';
import type { Request } from '../server/index.ts';

test('API validates, trains, generates, stops, and resumes a model', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-api-'));
  process.env.SEED_CHECKPOINT = path.join(directory, 'checkpoint.pt');
  process.env.SEED_DEVICE = 'cpu';
  const { handler, studio } = await import('../server/index.ts');
  // Exercise the real HTTP handler without binding a port (restricted in CI).

  const request = async <T = StudioStatus>(
    route: string,
    body?: unknown,
    headers: Record<string, string> = {},
  ) => {
    const req: Request = Object.assign(
      Readable.from(body === undefined ? [] : [Buffer.from(JSON.stringify(body))]),
      {
        url: '/api/' + route,
        method: body === undefined ? 'GET' : 'POST',
        headers: {
          host: '127.0.0.1:3001',
          ...Object.fromEntries(Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v])),
        },
      },
    );
    let status = 0;
    let output = '';
    await handler(req, {
      writeHead(code) {
        status = code;
      },

      end(data) {
        output = data.toString();
      },
    });
    return { status, data: JSON.parse(output) as T };
  };
  try {
    assert.equal((await request('status')).data.model, null);
    assert.equal((await request('generate', { prompt: 'a' })).status, 400);
    assert.equal(
      (await request('chat', { messages: [{ role: 'user', content: 'Hello!' }] })).status,
      400,
    );
    assert.equal((await request('train', { text: 'a' })).status, 400);
    assert.equal(
      (
        await request(
          'train',
          { text: 'abc'.repeat(50), steps: 3, hiddenSize: 16 },
          { Origin: 'https://example.com' },
        )
      ).status,
      403,
    );
    const started = await request('train', { text: 'abc'.repeat(50), steps: 20, hiddenSize: 16 });
    assert.equal(started.status, 202);
    const deadline = Date.now() + 5000;
    let current;
    do {
      await new Promise((resolve) => setTimeout(resolve, 15));
      current = (await request('status')).data;
    } while (current.running && Date.now() < deadline);
    assert.equal(current.running, false);
    assert.equal(current.model!.steps, 20);
    assert.ok(current.job!.loss! > 0);
    assert.equal((await request('generate', { prompt: '🦄', maxTokens: 1 })).status, 200);
    assert.equal((await request('generate', { prompt: 'a', temperature: 0 })).status, 400);
    const result = await request<GenerateResponse>('generate', { prompt: 'a', maxTokens: 30 });
    assert.equal(result.status, 200);
    assert.ok(result.data.tokens >= 1 && result.data.tokens <= 30);
    assert.equal(typeof result.data.completion, 'string');
    const resumed = await request('train', { text: 'abc'.repeat(50), resume: true, steps: 20000 });
    assert.equal(resumed.status, 202);
    assert.equal((await request('train', { text: 'abc'.repeat(50) })).status, 400);
    const stopped = await request('stop', {});
    assert.equal(stopped.data.running, false);
    const step = stopped.data.model!.steps;
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.equal((await request('status')).data.model!.steps, step);
    await request('train', { text: 'abc'.repeat(50), steps: 20000, hiddenSize: 16 });
    await request('stop', {});
    await request('train', { text: 'abc'.repeat(50), steps: 3, hiddenSize: 16 });
    const restartDeadline = Date.now() + 5000;
    while ((await request('status')).data.running && Date.now() < restartDeadline) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    const restarted = (await request('status')).data;
    assert.equal(
      restarted.model!.steps,
      3,
      'A stopped callback must not train the replacement model',
    );
    assert.equal(restarted.running, false);
    await request('train', {
      text: 'User: Hello!\nAssistant: Hello there.\n\n'.repeat(10),
      steps: 1,
      hiddenSize: 16,
    });
    const chatDeadline = Date.now() + 5000;
    while ((await request('status')).data.running && Date.now() < chatDeadline) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    const chatResponse = await request<ChatResponse>('chat', {
      messages: [{ role: 'user', content: 'Hello!' }],
      maxTokens: 40,
    });
    assert.equal(chatResponse.status, 200);
    assert.equal(typeof chatResponse.data.reply, 'string');
    assert.equal(chatResponse.data.modelStep, 1);
    assert.ok(chatResponse.data.tokens >= 1 && chatResponse.data.tokens <= 40);
    assert.equal((await request('chat', { messages: [] })).status, 400);
  } finally {
    try {
      await request('stop', {});
    } finally {
      await studio.close();
      await rm(directory, { recursive: true, force: true });
    }
  }
});
