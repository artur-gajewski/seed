import test from 'node:test';
import assert from 'node:assert/strict';
import { modelStats } from '../src/lib/modelStats.ts';
import type { StudioStatus } from '../shared/api.ts';

const model: NonNullable<StudioStatus['model']> = {
  architecture: 'transformer',
  hiddenSize: 128,
  layers: 3,
  heads: 4,
  contextSize: 256,
  vocabulary: 258,
  tokenizer: 'utf8-bytes',
  parameters: 694146,
  steps: 6000,
};

test('model stats show no architecture or update count before a model is loaded', () => {
  const stats = modelStats({ model: null, job: null, running: false });
  assert.equal(stats.architecture, '—');
  assert.equal(stats.parameters, '—');
  assert.equal(stats.vocabulary, '—');
  assert.equal(stats.steps, '—');
});

test('loading a different checkpoint updates architecture, parameters, vocabulary and total steps', () => {
  const loaded = modelStats({ model, job: null, running: false });
  assert.equal(loaded.parameters, (694146).toLocaleString());
  assert.match(loaded.architectureDetail, /Width 128 · 3 layers · 4 heads · 256-token context/);
  assert.equal(loaded.vocabulary, '258');
  assert.equal(loaded.steps, (6000).toLocaleString());
  const legacy = modelStats({
    model: {
      ...model,
      architecture: 'rnn',
      hiddenSize: 32,
      parameters: 12288,
      vocabulary: 40,
      tokenizer: 'characters',
      steps: 50,
    },
    job: null,
    running: false,
  });
  assert.equal(legacy.architecture, 'Character RNN');
  assert.equal(legacy.parameters, (12288).toLocaleString());
  assert.equal(legacy.vocabulary, '40');
  assert.equal(legacy.vocabularyUnit, 'characters');
  assert.equal(legacy.steps, '50');
});

test('resumed training displays global model steps separately from current session progress', () => {
  const job = {
    completed: 25,
    total: 1000,
    startedAt: 0,
    loss: 1,
    tokens: 1000,
    sequenceLength: 256,
    learningRate: 0.001,
  };
  const training = modelStats({ model: { ...model, steps: 6025 }, job, running: true });
  assert.equal(training.steps, (6025).toLocaleString());
  assert.equal(training.stepsDetail, `Training: 25 / ${(1000).toLocaleString()} updates`);
  assert.equal(training.parameters, model.parameters.toLocaleString());
  const stopped = modelStats({ model: { ...model, steps: 6025 }, job, running: false });
  assert.equal(stopped.steps, training.steps);
  assert.equal(stopped.stepsDetail, `Session: 25 / ${(1000).toLocaleString()} updates`);
});

test('selected settings estimate the actual Transformer size independently of batch size', async () => {
  const { selectedModelStats } = await import('../src/lib/modelStats.ts');
  const settings = {
    hiddenSize: 512,
    layers: 4,
    heads: 8,
    sequenceLength: 256,
    batchSize: 8,
    steps: 6000,
    learningRate: 0.001,
  };
  assert.equal(selectedModelStats(settings).parameters, 13006082);
  assert.equal(selectedModelStats({ ...settings, batchSize: 1 }).parameters, 13006082);
  assert.equal(
    selectedModelStats({ ...settings, hiddenSize: 128, layers: 3, heads: 4 }).parameters,
    694146,
  );
});
