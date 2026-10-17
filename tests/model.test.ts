import test from 'node:test';
import assert from 'node:assert/strict';
import { LanguageModel, parameterKeys } from '../server/legacy-model.ts';

test('manual backpropagation matches numerical derivatives for every parameter group', () => {
  const model = new LanguageModel('abc', 4);
  const input = [0, 1, 2, 0];
  const target = [1, 2, 0, 1];
  const { gradients } = model.lossAndGradients(input, target);
  for (const key of parameterKeys) {
    for (let i = 0; i < model.weights[key].length; i++) {
      const old = model.weights[key][i];
      const epsilon = 1e-5;
      model.weights[key][i] = old + epsilon;
      const plus = model.lossAndGradients(input, target).loss;
      model.weights[key][i] = old - epsilon;
      const minus = model.lossAndGradients(input, target).loss;
      model.weights[key][i] = old;
      assert.ok(
        Math.abs((plus - minus) / (2 * epsilon) - gradients[key][i]) < 1e-7,
        `${key}[${i}] gradient mismatch`,
      );
    }
  }
});
test('training learns a repeating sequence from random weights', () => {
  const model = new LanguageModel('abc', 16);
  const input = model.encode('abcabcabcabc');
  const target = model.encode('bcabcabcabca');
  const before = model.lossAndGradients(input, target).loss;
  for (let i = 0; i < 180; i++) {
    model.train(input, target, 0.01);
  }
  const after = model.lossAndGradients(input, target).loss;
  assert.ok(after < before * 0.1, `${before} -> ${after}`);
  assert.equal(model.generate('a', 12, 0.1, 1), 'bcabcabcabca');
});
test('checkpoint preserves predictions and Adam state for continued training', () => {
  const model = new LanguageModel('a🌱b', 8);
  model.train([0, 1, 2], [1, 2, 0], 0.003);
  const loaded = LanguageModel.restore(JSON.parse(JSON.stringify(model.serialize())));
  assert.deepEqual(loaded.serialize(), model.serialize());
  loaded.train([0, 1, 2], [1, 2, 0], 0.003);
  model.train([0, 1, 2], [1, 2, 0], 0.003);
  assert.deepEqual(loaded.serialize(), model.serialize());
  assert.equal(Array.from(loaded.generate('🌱', 10)).length, 10);
  assert.throws(() => loaded.encode('z'), /outside the training vocabulary/);
});
test('softmax remains normalized with large logits', () => {
  const model = new LanguageModel('abc', 4);
  model.weights.outputBias.set([10000, 9999, -10000]);
  const { probabilities } = model.forward(0, new Float64Array(4), 0.1);
  assert.ok(probabilities.every(Number.isFinite));
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
});
