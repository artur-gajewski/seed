import { decodeCheckpoint } from '../server/checkpoint.ts';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { ModelStudio } from '../server/studio.ts';
import { LanguageModel } from '../server/model.ts';
import type { ModelCheckpoint } from '../server/model.ts';
import { LanguageModel as LegacyLanguageModel } from '../server/legacy-model.ts';

async function waitForTraining(studio: ModelStudio): Promise<void> {
  const deadline = Date.now() + 5000;

  while (studio.getStatus().running && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }

  assert.equal(studio.getStatus().running, false, 'Training should finish before the deadline');
  assert.equal(studio.getStatus().error, null);
}

test('studio saves, restores, and resumes version 2 Transformer checkpoints', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-studio-'));
  const checkpointPath = path.join(directory, 'data/checkpoint.json');
  const studio = new ModelStudio(checkpointPath);
  const restoredStudio = new ModelStudio(checkpointPath);
  const corpus = 'abc'.repeat(50);

  try {
    const settings = {
      hiddenSize: 16,
      steps: 10,
      layers: 1,
      heads: 2,
      sequenceLength: 16,
      learningRate: 0.002,
      batchSize: 3,
    };
    studio.startTraining({ text: corpus, ...settings });
    await waitForTraining(studio);
    const trainedStatus = studio.getStatus();

    await studio.saveCheckpoint();
    const saved = decodeCheckpoint(await readFile(checkpointPath)) as { model: ModelCheckpoint };
    assert.equal(saved.model.version, 2);
    assert.ok(saved.model.version === 2);
    assert.equal(saved.model.architecture, 'transformer');
    assert.ok(Number.isFinite(trainedStatus.job!.validationLoss));
    assert.equal(
      trainedStatus.job!.trainingTokens! + trainedStatus.job!.validationTokens!,
      trainedStatus.job!.tokens,
    );
    assert.equal(saved.model.step, 10);

    await restoredStudio.loadCheckpoint();
    assert.deepEqual(restoredStudio.getStatus().model, trainedStatus.model);
    assert.deepEqual(restoredStudio.getStatus().history, trainedStatus.history);
    assert.deepEqual(restoredStudio.getStatus().trainingSettings, settings);
    assert.equal(restoredStudio.getStatus().job, null);

    // Top K = 1 chooses the same most likely token regardless of the sampling RNG.
    const prompt = { prompt: 'a', maxTokens: 20, topK: 1 };
    assert.equal(restoredStudio.generate(prompt).completion, studio.generate(prompt).completion);

    restoredStudio.startTraining({ text: corpus, resume: true, steps: 3 });
    await assert.rejects(restoredStudio.loadCheckpoint(), /Stop training before loading/);
    await waitForTraining(restoredStudio);
    assert.equal(restoredStudio.getStatus().model!.steps, 13);

    await restoredStudio.saveCheckpoint();
    await studio.loadCheckpoint();
    assert.deepEqual(studio.getStatus().model, restoredStudio.getStatus().model);

    // Older files still supply architecture settings and clear settings from another model.
    await writeFile(checkpointPath, JSON.stringify({ model: saved.model }));
    await studio.loadCheckpoint();
    assert.deepEqual(studio.getStatus().trainingSettings, {
      hiddenSize: 16,
      layers: 1,
      heads: 2,
      sequenceLength: 16,
    });
  } finally {
    studio.stopTraining();
    restoredStudio.stopTraining();
    await rm(directory, { recursive: true, force: true });
  }
});

test('training automatically saves every 100 model updates across resumed jobs', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-autosave-'));
  const checkpointPath = path.join(directory, 'data/checkpoint.json');
  const studio = new ModelStudio(checkpointPath);
  const settings = {
    text: 'abc'.repeat(50),
    hiddenSize: 16,
    layers: 1,
    sequenceLength: 8,
    batchSize: 1,
  };

  try {
    studio.startTraining({ ...settings, steps: 99 });
    await waitForTraining(studio);
    await assert.rejects(readFile(checkpointPath), { code: 'ENOENT' });

    for (const steps of [2, 100]) {
      studio.startTraining({ ...settings, resume: true, steps });
      await waitForTraining(studio);
      const saved = decodeCheckpoint(await readFile(checkpointPath)) as {
        model: ModelCheckpoint;
        history: Array<{ step: number }>;
      };
      assert.equal(saved.model.step, studio.getStatus().model!.steps - 1);
      assert.equal(saved.history.at(-1)!.step, saved.model.step);
      const restored = new ModelStudio(checkpointPath);
      await restored.loadCheckpoint();
      assert.equal(restored.getStatus().model!.steps, saved.model.step);
    }
  } finally {
    studio.stopTraining();
    await rm(directory, { recursive: true, force: true });
  }
});

test('automatic checkpoint failures are reported in training status', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-autosave-error-'));
  const parentPath = path.join(directory, 'file');
  await writeFile(parentPath, 'not a directory');
  const studio = new ModelStudio(path.join(parentPath, 'checkpoint.json'));

  try {
    studio.startTraining({
      text: 'abc'.repeat(50),
      hiddenSize: 16,
      layers: 1,
      sequenceLength: 8,
      batchSize: 1,
      steps: 101,
    });
    const deadline = Date.now() + 5000;
    while (studio.getStatus().running && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.equal(studio.getStatus().running, false);
    assert.equal(studio.getStatus().model!.steps, 100);
    assert.match(studio.getStatus().error!, /EEXIST|ENOTDIR/);
  } finally {
    studio.stopTraining();
    await rm(directory, { recursive: true, force: true });
  }
});

test('deleting a checkpoint removes only the saved file and allows saving again', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-delete-'));
  const checkpointPath = path.join(directory, 'checkpoint.json');
  const studio = new ModelStudio(checkpointPath);

  try {
    await studio.deleteCheckpoint();
    assert.equal(await studio.getCheckpointInfo(), null);
    studio.startTraining({ text: 'abc'.repeat(50), hiddenSize: 16, steps: 1 });
    await waitForTraining(studio);
    await studio.saveCheckpoint();
    const fileInfo = await stat(checkpointPath);
    const expectedInfo = { savedAt: fileInfo.mtime.toISOString(), sizeBytes: fileInfo.size };
    assert.deepEqual(await studio.getCheckpointInfo(), expectedInfo);
    assert.deepEqual(
      await new ModelStudio(checkpointPath).getCheckpointInfo(),
      expectedInfo,
      'Existing checkpoint metadata is available without loading a model',
    );
    const status = studio.getStatus();
    const prompt = { prompt: 'a', maxTokens: 5, topK: 1 };
    const completion = studio.generate(prompt).completion;
    await studio.deleteCheckpoint();
    assert.equal(await studio.getCheckpointInfo(), null);
    await assert.rejects(readFile(checkpointPath), { code: 'ENOENT' });
    assert.deepEqual(studio.getStatus(), status);
    assert.equal(studio.generate(prompt).completion, completion);
    await studio.deleteCheckpoint();
    await studio.saveCheckpoint();
    await studio.loadCheckpoint();
    assert.deepEqual(studio.getStatus().model, status.model);
  } finally {
    studio.stopTraining();
    await rm(directory, { recursive: true, force: true });
  }
});

test('version 1 RNN checkpoints still load and retain their model identity', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-legacy-'));
  const checkpointPath = path.join(directory, 'checkpoint.json');
  const legacy = new LegacyLanguageModel('abc', 16);
  legacy.train([0, 1, 2], [1, 2, 0], 0.003);
  const restored = LanguageModel.restore(legacy.serialize());
  assert.ok(restored instanceof LegacyLanguageModel);
  assert.deepEqual(restored.serialize(), legacy.serialize());
  const studio = new ModelStudio(checkpointPath);

  try {
    await writeFile(checkpointPath, JSON.stringify({ model: legacy.serialize(), history: [] }));
    await studio.loadCheckpoint();
    assert.equal(studio.getStatus().model!.architecture, 'rnn');
    assert.equal(
      studio.generate({ prompt: 'a', topK: 1, maxTokens: 12 }).completion,
      legacy.generate('a', 12, 0.8, 1),
    );
    await studio.saveCheckpoint();
    const saved = decodeCheckpoint(await readFile(checkpointPath)) as { model: ModelCheckpoint };
    assert.equal(saved.model.version, 1);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
