import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_TRAINING_SETTINGS } from '../src/lib/trainingSettings.ts';
import {
  DEFAULT_GENERATION_SETTINGS,
  isTrainingSettings,
  isGenerationSettings,
  isTrainingDevice,
  readPreference,
  writePreference,
} from '../src/lib/studioPreferences.ts';
import { parseTrainingRequest, parseGenerationRequest } from '../server/validation.ts';

test('maximum defaults satisfy the API contracts', () => {
  assert.deepEqual(parseTrainingRequest({ text: 'abc'.repeat(30), ...DEFAULT_TRAINING_SETTINGS }), {
    text: 'abc'.repeat(30),
    ...DEFAULT_TRAINING_SETTINGS,
    resume: false,
  });
  assert.deepEqual(
    parseGenerationRequest({ ...DEFAULT_GENERATION_SETTINGS }),
    DEFAULT_GENERATION_SETTINGS,
  );
});

test('preferences survive visits and reject corrupt or invalid values without losing valid settings', () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  const values = new Map<string, string>();
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    },
  });
  try {
    assert.deepEqual(
      readPreference('train', DEFAULT_TRAINING_SETTINGS, isTrainingSettings),
      DEFAULT_TRAINING_SETTINGS,
    );
    const training = { ...DEFAULT_TRAINING_SETTINGS, steps: 500, learningRate: 0.001 };
    const generation = {
      ...DEFAULT_GENERATION_SETTINGS,
      prompt: 'Hello 🌱',
      temperature: 0.8,
      topK: 0,
    };
    writePreference('train', training, isTrainingSettings);
    writePreference('generate', generation, isGenerationSettings);
    writePreference('device', 'mps', isTrainingDevice);
    assert.deepEqual(
      readPreference('train', DEFAULT_TRAINING_SETTINGS, isTrainingSettings),
      training,
    );
    assert.deepEqual(
      readPreference('generate', DEFAULT_GENERATION_SETTINGS, isGenerationSettings),
      generation,
    );
    assert.equal(readPreference('device', 'auto', isTrainingDevice), 'mps');
    writePreference('train', { ...training, steps: 0 }, isTrainingSettings);
    assert.deepEqual(
      readPreference('train', DEFAULT_TRAINING_SETTINGS, isTrainingSettings),
      training,
    );
    for (const invalid of [
      '{broken',
      'null',
      '[]',
      JSON.stringify({ ...training, heads: 3 }),
      JSON.stringify({ ...training, sequenceLength: 2049 }),
    ]) {
      values.set('train', invalid);
      assert.deepEqual(
        readPreference('train', DEFAULT_TRAINING_SETTINGS, isTrainingSettings),
        DEFAULT_TRAINING_SETTINGS,
      );
    }
    values.set('generate', JSON.stringify({ ...generation, maxTokens: 1001 }));
    assert.deepEqual(
      readPreference('generate', DEFAULT_GENERATION_SETTINGS, isGenerationSettings),
      DEFAULT_GENERATION_SETTINGS,
    );
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      get() {
        throw new Error('Storage unavailable');
      },
    });
    assert.deepEqual(
      readPreference('train', DEFAULT_TRAINING_SETTINGS, isTrainingSettings),
      DEFAULT_TRAINING_SETTINGS,
    );
    assert.doesNotThrow(() => writePreference('train', training, isTrainingSettings));
  } finally {
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
});
