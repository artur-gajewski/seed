import test from 'node:test';
import assert from 'node:assert/strict';
import { LanguageModel } from '../server/model.ts';

function smallModel(): LanguageModel {
  return new LanguageModel('', 8, 42, { layers: 1, heads: 2, contextSize: 8 });
}

test('Transformer derivatives match numerical gradients across every parameter group', () => {
  const model = smallModel();
  const inputs = model.encode('abc');
  const targets = model.encode('bca');
  const { gradients } = model.lossAndGradients(inputs, targets);
  for (const [name, parameter] of Object.entries(model.weights)) {
    const indices = new Set([
      0,
      1,
      Math.floor(parameter.values.length / 2),
      parameter.values.length - 1,
    ]);
    if (name === 'tokenEmbedding')
      for (let index = 97 * 8; index < 100 * 8; index++) indices.add(index);
    for (const index of indices) {
      const original = parameter.values[index];
      const epsilon = 1e-5;
      parameter.values[index] = original + epsilon;
      const plus = model.evaluate(inputs, targets);
      parameter.values[index] = original - epsilon;
      const minus = model.evaluate(inputs, targets);
      parameter.values[index] = original;
      assert.ok(
        Math.abs((plus - minus) / (2 * epsilon) - gradients[name][index]) < 2e-5,
        `${name}[${index}] gradient mismatch`,
      );
    }
  }
});

test('causal attention prevents future tokens from changing earlier predictions', () => {
  const model = smallModel();
  const first = model.forward(model.encode('abc'));
  const changedFuture = model.forward(model.encode('azz'));
  assert.deepEqual(
    first.values.slice(0, model.vocabularySize),
    changedFuture.values.slice(0, model.vocabularySize),
  );
});

test('UTF-8 tokenization accepts characters absent from the corpus', () => {
  const model = smallModel();
  assert.equal(Buffer.from(model.encode('Hello 🌱, moi!')).toString('utf8'), 'Hello 🌱, moi!');
  assert.equal(model.vocabularySize, 258);
});

test('Transformer learns next-token patterns from random weights', () => {
  const model = new LanguageModel('', 16, 42, { layers: 1, heads: 2, contextSize: 8 });
  const tokens = model.encode('abc'.repeat(8));
  const before = model.evaluate(tokens.slice(0, 8), tokens.slice(1, 9));
  for (let step = 0; step < 220; step++) {
    const offset = step % 3;
    model.train(tokens.slice(offset, offset + 8), tokens.slice(offset + 1, offset + 9), 0.005);
  }
  const after = model.evaluate(tokens.slice(0, 8), tokens.slice(1, 9));
  assert.ok(
    after < 0.1 && after < before / 10,
    `Loss did not improve enough: ${before} -> ${after}`,
  );
  assert.equal(model.generate('abcabcab', 12, 0.1, 1), 'cabcabcabcab');
});

test('Transformer checkpoints preserve configuration and exact resumed AdamW updates', () => {
  const model = smallModel();
  model.train(model.encode('abc'), model.encode('bca'), 0.001);
  const loaded = LanguageModel.restore(model.serialize());
  assert.ok(loaded instanceof LanguageModel);
  assert.deepEqual(loaded.serialize(), model.serialize());
  loaded.train(model.encode('abc'), model.encode('bca'), 0.001);
  model.train(model.encode('abc'), model.encode('bca'), 0.001);
  assert.deepEqual(loaded.serialize(), model.serialize());
  const corrupted = model.serialize();
  corrupted.parameters.outputBias.values[0] = Infinity;
  assert.throws(() => LanguageModel.restore(corrupted), /Invalid checkpoint/);
});

test('a small Transformer learns a dialogue response and its end marker from scratch', () => {
  const model = new LanguageModel('', 16, 42, { layers: 1, heads: 2, contextSize: 64 });
  const tokens = model.encodeTrainingText('User: Hello!\nAssistant: Hello!\n\n');
  for (let step = 0; step < 500; step++) {
    model.train(tokens.slice(0, -1), tokens.slice(1), 0.003);
  }
  const result = model.generateDetailed('User: Hello!\nAssistant:', 50, 0.1, 1);
  assert.equal(result.completion, ' Hello!');
  assert.equal(result.tokens, 8, 'Seven answer bytes plus the learned end marker');
});

test('end-of-response tokens stop generation and are never displayed as text', () => {
  const model = smallModel();
  model.weights.outputBias.values[257] = 1000;
  assert.deepEqual(model.generateDetailed('Hello!', 50, 0.1, 1), { completion: '', tokens: 1 });
});
