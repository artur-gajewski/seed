import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { TorchStudio } from '../server/torch-studio.ts';
import { LanguageModel } from '../server/model.ts';
import { encodeCheckpoint } from '../server/checkpoint.ts';

const text = 'User: What color is a red car?\nAssistant: Red.\n\n'.repeat(8);
async function finished(studio: TorchStudio) {
  const deadline = Date.now() + 15000;
  let status = await studio.getStatus();
  while (status.running && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 20));
    status = await studio.getStatus();
  }
  assert.equal(status.error, null);
  assert.equal(status.running, false);
  return status;
}

test('PyTorch studio persists training, resumes CLI checkpoints, and serves status during training', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-studio-'));
  const file = path.join(directory, 'checkpoint.pt');
  const studio = new TorchStudio(file, { device: 'cpu' });
  const restored = new TorchStudio(file, { device: 'cpu' });
  try {
    const initial = await studio.getStatus();
    assert.equal(initial.backend?.engine, 'pytorch');
    assert.equal(initial.backend.device, 'cpu');
    assert.equal(initial.trainingSettings, undefined);
    await studio.startTraining({
      text,
      hiddenSize: 16,
      layers: 1,
      heads: 2,
      sequenceLength: 16,
      steps: 3,
    });
    const complete = await finished(studio);
    assert.equal(complete.model!.steps, 3);
    assert.ok(complete.job!.validationLoss! > 0);
    assert.equal(complete.model!.checkpointFileName, 'checkpoint.pt');
    assert.ok((await readFile(file)).length > 0);
    const loaded = await restored.loadCheckpoint();
    assert.equal(loaded.model!.steps, 3);
    assert.deepEqual(loaded.history, complete.history);
    const generated = await restored.generate({ prompt: 'Hi', maxTokens: 3, topK: 1 });
    assert.ok(generated.tokens >= 1 && generated.tokens <= 3);
    await restored.startTraining({ text, resume: true, steps: 20000 });
    const started = performance.now();
    assert.equal((await restored.getStatus()).running, true);
    assert.ok(performance.now() - started < 1000, 'Status must not wait for training');
    await assert.rejects(restored.loadCheckpoint(), /Stop training/);
    const stopped = await restored.stopTraining();
    assert.equal(stopped.running, false);
    const step = stopped.model!.steps;
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.equal((await restored.getStatus()).model!.steps, step);
    await assert.rejects(restored.loadCheckpoint({ fileName: '../other.pt' }), /filename/);
    await restored.deleteCheckpoint();
    assert.equal((await restored.getStatus()).checkpoint, null);
    assert.equal((await restored.getStatus()).model!.steps, step);
    await restored.saveCheckpoint();
    assert.ok((await restored.getStatus()).checkpoint);
  } finally {
    await Promise.all([studio.close(), restored.close()]);
    await rm(directory, { recursive: true, force: true });
  }
});

test('PyTorch loads legacy binary Transformer checkpoints without changing the source', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-migration-'));
  const studio = new TorchStudio(path.join(directory, 'checkpoint.pt'), { device: 'cpu' });
  try {
    const model = new LanguageModel('', 16, 42, { layers: 1, heads: 2, contextSize: 16 });
    model.train([256, 65, 66], [65, 66, 257], 0.001);
    const source = encodeCheckpoint({ model: model.serialize(), history: [{ step: 1, loss: 2 }] });
    await writeFile(path.join(directory, 'checkpoint.json'), source);
    const loaded = await studio.loadCheckpoint();
    assert.equal(loaded.model!.steps, 1);
    assert.equal(loaded.history[0].loss, 2);
    assert.deepEqual(await readFile(path.join(directory, 'checkpoint.json')), source);
    await writeFile(path.join(directory, 'broken.pt'), 'corrupt');
    await assert.rejects(studio.loadCheckpoint({ fileName: 'broken.pt' }));
    assert.equal((await studio.getStatus()).model!.steps, 1);
    await studio.startTraining({ text, resume: true, steps: 1, sequenceLength: 16 });
    assert.equal((await finished(studio)).model!.steps, 2);
  } finally {
    await studio.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test('PyTorch New life preserves corrections, isolates presets, and restores lessons', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-life-'));
  const file = path.join(directory, 'checkpoint.pt');
  const studio = new TorchStudio(file, { device: 'cpu' });
  const restored = new TorchStudio(file, { device: 'cpu' });
  try {
    assert.deepEqual(await studio.life('status'), {
      ready: false,
      steps: 0,
      lessons: 0,
      corrections: 0,
    });
    const taught = await studio.life('teach', { message: 'Hi', answer: 'Hello' });
    assert.deepEqual(taught, { ready: true, steps: 12, lessons: 1, corrections: 1 });
    const reply = await studio.life('message', { message: 'Hi' });
    assert.ok('reply' in reply);
    assert.equal(reply.status.steps, 24);
    assert.equal(reply.status.corrections, 1);
    assert.deepEqual(await restored.life('status'), reply.status);
    assert.deepEqual(await studio.life('status', { size: 'medium' }), {
      ready: false,
      steps: 0,
      lessons: 0,
      corrections: 0,
    });
    await assert.rejects(
      studio.life('teach', { message: 'Hi\nAssistant:', answer: 'bad' }),
      /one line/,
    );
    const learning = studio.life('teach', { message: 'Sky?', answer: 'Blue' });
    await assert.rejects(studio.life('message', { message: 'Hi' }), /learning/);
    await learning;
    // A failed atomic replacement must leave the in-memory life unchanged.
    await rm(path.join(directory, 'new-life-checkpoint-38K.pt'));
    await mkdir(path.join(directory, 'new-life-checkpoint-38K.pt'));
    const before = await studio.life('status');
    await assert.rejects(studio.life('teach', { message: 'New', answer: 'Lesson' }));
    assert.deepEqual(await studio.life('status'), before);
  } finally {
    await Promise.all([studio.close(), restored.close()]);
    await rm(directory, { recursive: true, force: true });
  }
});

test('missing Python is reported without hanging requests', async () => {
  const studio = new TorchStudio('/tmp/seed-unused.pt', { python: '/missing/seed-python' });
  try {
    await assert.rejects(studio.getStatus(), /Cannot start PyTorch/);
    assert.match((await studio.getStatus()).error!, /Cannot start PyTorch/);
    await assert.rejects(studio.generate({ prompt: 'Hi' }), /Cannot start PyTorch/);
  } finally {
    await studio.close();
  }
});

test('PyTorch New life migrates legacy lessons and rejects incompatible checkpoints before writing', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-life-migration-'));
  const studio = new TorchStudio(path.join(directory, 'checkpoint.pt'), { device: 'cpu' });
  try {
    const model = new LanguageModel('', 32, 42, { layers: 1, heads: 4, contextSize: 256 });
    const legacy = path.join(directory, 'new-life-checkpoint.json');
    const source = encodeCheckpoint({
      version: 1,
      model: model.serialize(),
      lessons: [{ message: 'Hi', answer: 'Hello' }],
    });
    await writeFile(legacy, source);
    assert.deepEqual(await studio.life('status'), {
      ready: true,
      steps: 0,
      lessons: 1,
      corrections: 1,
    });
    assert.deepEqual(await readFile(legacy), source);
    assert.ok((await readFile(path.join(directory, 'new-life-checkpoint-38K.pt'))).length > 0);
    await writeFile(path.join(directory, 'new-life-checkpoint-medium.json'), source);
    await assert.rejects(studio.life('status', { size: 'medium' }), /selected model size/);
    await assert.rejects(readFile(path.join(directory, 'new-life-checkpoint-496K.pt')), {
      code: 'ENOENT',
    });
  } finally {
    await studio.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test('overlapping checkpoint changes cannot change the destination of a training request', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-mutations-'));
  const studio = new TorchStudio(path.join(directory, 'checkpoint.pt'), { device: 'cpu' });
  try {
    const loading = studio.loadCheckpoint({ fileName: 'missing.pt' });
    await assert.rejects(
      studio.startTraining({ text, steps: 1 }),
      /operation is already in progress/,
    );
    await assert.rejects(loading, { code: 'ENOENT' });
    assert.equal((await studio.getStatus()).model, null);
  } finally {
    await studio.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test('a worker crash rejects pending requests and remains visible in status', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-worker-crash-'));
  const executable = path.join(directory, 'fake-python');
  await writeFile(
    executable,
    `#!/bin/sh
printf '%s\\n' '{"event":"ready","version":1,"status":{"running":false,"model":null,"job":null,"history":[],"error":null}}'
read -r request
exit 3
`,
    { mode: 0o755 },
  );
  const studio = new TorchStudio(path.join(directory, 'checkpoint.pt'), { python: executable });
  try {
    await studio.getStatus();
    await assert.rejects(studio.generate({ prompt: 'Hi' }), /worker exited/);
    assert.match((await studio.getStatus()).error!, /worker exited/);
  } finally {
    await studio.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test('graceful worker shutdown saves a running training job', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-shutdown-'));
  const file = path.join(directory, 'checkpoint.pt');
  const studio = new TorchStudio(file, { device: 'cpu' });
  const restored = new TorchStudio(file, { device: 'cpu' });
  try {
    await studio.startTraining({
      text,
      hiddenSize: 16,
      layers: 1,
      heads: 2,
      sequenceLength: 16,
      steps: 20000,
    });
    let status = await studio.getStatus();
    const deadline = Date.now() + 10000;
    while (!status.job?.completed && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 20));
      status = await studio.getStatus();
    }
    assert.ok(status.job!.completed > 0);
    await studio.close();
    const loaded = await restored.loadCheckpoint();
    assert.ok(loaded.model!.steps >= status.model!.steps);
    assert.equal(loaded.running, false);
  } finally {
    await Promise.all([studio.close(), restored.close()]);
    await rm(directory, { recursive: true, force: true });
  }
});
