import test from 'node:test';
import assert from 'node:assert/strict';
import type { StudioStatus } from '../shared/api.ts';
import {
  DEFAULT_TRAINING_SETTINGS,
  trainingSettingsForModel,
} from '../src/lib/trainingSettings.ts';

const loaded: StudioStatus = {
  model: {
    hiddenSize: 512,
    layers: 4,
    heads: 8,
    contextSize: 256,
    vocabulary: 258,
    parameters: 13006082,
    steps: 12000,
    architecture: 'transformer',
  },
  trainingSettings: {
    hiddenSize: 128,
    layers: 3,
    heads: 4,
    sequenceLength: 256,
    batchSize: 8,
    steps: 6000,
    learningRate: 0.001,
  },
  running: false,
  job: null,
  history: [],
  error: null,
};

test('loading a checkpoint prefills its actual architecture and saved session settings', () => {
  assert.deepEqual(trainingSettingsForModel(loaded), {
    hiddenSize: 512,
    layers: 4,
    heads: 8,
    sequenceLength: 256,
    batchSize: 8,
    steps: 6000,
    learningRate: 0.001,
  });
  assert.equal(loaded.trainingSettings!.hiddenSize, 128, 'Do not mutate API state');
});

test('checkpoints without training metadata use model configuration and default session settings', () => {
  const settings = trainingSettingsForModel({ ...loaded, trainingSettings: undefined });
  assert.equal(settings!.hiddenSize, 512);
  assert.equal(settings!.sequenceLength, 256);
  assert.equal(settings!.steps, DEFAULT_TRAINING_SETTINGS.steps);
  assert.equal(settings!.learningRate, DEFAULT_TRAINING_SETTINGS.learningRate);
});

test('restored sequence length retains a shorter training window and never exceeds model capacity', () => {
  assert.equal(
    trainingSettingsForModel({ ...loaded, trainingSettings: { sequenceLength: 64 } })!
      .sequenceLength,
    64,
  );
  assert.equal(
    trainingSettingsForModel({ ...loaded, model: { ...loaded.model!, contextSize: 128 } })!
      .sequenceLength,
    128,
  );
  assert.equal(trainingSettingsForModel({ ...loaded, model: null }), null);
});
