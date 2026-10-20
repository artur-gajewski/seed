import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { gelu, linear, tensor } from '../server/tensor.ts';
import type { Tensor } from '../server/tensor.ts';

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-11) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++)
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance * (1 + Math.abs(expected[index])),
      `Mismatch at ${index}: ${actual[index]} vs ${expected[index]}`,
    );
}

// Original scalar kernel, kept independent of the accelerated implementation.
function referenceLinear(input: Tensor, weights: Tensor, bias: Tensor) {
  const output = tensor(input.rows, weights.columns);
  for (let row = 0; row < input.rows; row++)
    for (let column = 0; column < weights.columns; column++) {
      let value = bias.values[column];
      for (let feature = 0; feature < input.columns; feature++)
        value +=
          input.values[row * input.columns + feature] *
          weights.values[feature * weights.columns + column];
      output.values[row * output.columns + column] = value;
    }
  return {
    output,
    backward() {
      for (let row = 0; row < input.rows; row++)
        for (let column = 0; column < weights.columns; column++) {
          const gradient = output.gradients[row * output.columns + column];
          bias.gradients[column] += gradient;
          for (let feature = 0; feature < input.columns; feature++) {
            input.gradients[row * input.columns + feature] +=
              weights.values[feature * weights.columns + column] * gradient;
            weights.gradients[feature * weights.columns + column] +=
              input.values[row * input.columns + feature] * gradient;
          }
        }
    },
  };
}

test('cached GELU matches the original formula and derivatives, including saturation', () => {
  const values = [-20, -5, -1, 0, 1, 5, 20];
  const input = tensor(1, values.length, Float64Array.from(values));
  const tape: Array<() => void> = [];
  const output = gelu(input, tape);
  output.gradients.fill(1);
  tape[0]();
  const factor = Math.sqrt(2 / Math.PI);
  close(
    output.values,
    values.map((v) => 0.5 * v * (1 + Math.tanh(factor * (v + 0.044715 * v ** 3)))),
  );
  close(
    input.gradients,
    values.map((v) => {
      const activation = Math.tanh(factor * (v + 0.044715 * v ** 3));
      return (
        0.5 * (1 + activation) +
        0.5 * v * (1 - activation * activation) * factor * (1 + 3 * 0.044715 * v * v)
      );
    }),
  );
  close(gelu(input, null).values, output.values);
});

test('linear matches original outputs and accumulated gradients for rectangular shapes', () => {
  for (const [rows, features, columns] of [
    [3, 7, 9],
    [1, 4, 12],
    [8, 16, 4],
  ]) {
    const make = (r: number, c: number) => {
      const value = tensor(
        r,
        c,
        Float64Array.from({ length: r * c }, (_, i) => Math.sin(i + 0.5)),
      );
      value.gradients.fill(0.125);
      return value;
    };
    const x = make(rows, features),
      w = make(features, columns),
      b = make(1, columns);
    const rx = make(rows, features),
      rw = make(features, columns),
      rb = make(1, columns);
    const tape: Array<() => void> = [];
    const output = linear(x, w, b, tape);
    const reference = referenceLinear(rx, rw, rb);
    close(output.values, reference.output.values);
    for (let repeat = 0; repeat < 2; repeat++) {
      output.gradients.set(Float64Array.from(output.values, (_, i) => Math.cos(i + repeat)));
      reference.output.gradients.set(output.gradients);
      tape[0]();
      reference.backward();
      close(x.gradients, rx.gradients);
      close(w.gradients, rw.gradients);
      close(b.gradients, rb.gradients);
    }
  }
});

test('backend selection supports explicit JavaScript and rejects invalid or unsupported backends', () => {
  const source = `import { tensorBackend } from ${JSON.stringify(new URL('../server/numerics.ts', import.meta.url).href)}; console.log(tensorBackend);`;
  const run = (backend: string) =>
    spawnSync(process.execPath, ['--input-type=module', '-e', source], {
      env: { ...process.env, SEED_TENSOR_BACKEND: backend },
      encoding: 'utf8',
    });
  const portable = run('javascript');
  assert.equal(portable.status, 0, portable.stderr);
  assert.equal(portable.stdout.trim(), 'javascript');
  const invalid = run('unknown');
  assert.notEqual(invalid.status, 0);
  assert.match(invalid.stderr, /SEED_TENSOR_BACKEND must be/);
  const unsupported = run(process.platform === 'darwin' ? 'openblas' : 'accelerate');
  assert.notEqual(unsupported.status, 0);
  assert.match(unsupported.stderr, /is unavailable/);
});

const nativeBackend = process.platform === 'darwin' ? 'accelerate' : 'openblas';
const addon = new URL(`../native/${nativeBackend}.node`, import.meta.url);
const hasNative = ['darwin', 'linux'].includes(process.platform) && existsSync(addon);
test(
  `${nativeBackend} matches JavaScript forward, gradients, AdamW moments, and resumed updates`,
  {
    skip: !hasNative,
  },
  () => {
    const source = `
    import { LanguageModel } from ${JSON.stringify(new URL('../server/model.ts', import.meta.url).href)};
    import { tensorBackend } from ${JSON.stringify(new URL('../server/numerics.ts', import.meta.url).href)};
    if (tensorBackend !== process.env.SEED_TENSOR_BACKEND) throw new Error('Requested backend was not selected');
    const model = new LanguageModel('', 8, 42, { layers: 2, heads: 2, contextSize: 8 });
    const batch = [{ inputs: [97,98,99,97], targets: [98,99,97,98] },
                   { inputs: [99,98,97,99], targets: [98,97,99,98] }];
    const logits = Array.from(model.forward(batch[0].inputs).values);
    const result = model.lossAndGradients(batch[0].inputs, batch[0].targets);
    const gradients = Object.fromEntries(Object.entries(result.gradients).map(([k,v]) => [k,Array.from(v)]));
    const losses = [model.trainBatch(batch, 0.0003), model.trainBatch(batch, 0.0003)];
    const resumed = LanguageModel.restore(model.serialize());
    losses.push(resumed.trainBatch(batch, 0.0003));
    console.log(JSON.stringify({logits, gradients, loss:result.loss, losses, checkpoint:resumed.serialize()}));
  `;
    const run = (backend: string) => {
      const result = spawnSync(process.execPath, ['--input-type=module', '-e', source], {
        env: { ...process.env, SEED_TENSOR_BACKEND: backend },
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024,
      });
      assert.equal(result.status, 0, result.stderr);
      return JSON.parse(result.stdout);
    };
    const compare = (actual: unknown, expected: unknown): void => {
      if (typeof actual === 'number' && typeof expected === 'number')
        close([actual], [expected], 1e-10);
      else if (actual && expected && typeof actual === 'object' && typeof expected === 'object') {
        assert.deepEqual(Object.keys(actual), Object.keys(expected));
        for (const key of Object.keys(actual))
          compare(
            (actual as Record<string, unknown>)[key],
            (expected as Record<string, unknown>)[key],
          );
      } else assert.deepEqual(actual, expected);
    };
    compare(run(nativeBackend), run('javascript'));
  },
);

test(
  'native matrix multiplication validates buffers and rejects overlapping output',
  { skip: !hasNative },
  () => {
    const { matmul } = createRequire(import.meta.url)(fileURLToPath(addon));
    const a = new Float64Array(6),
      b = new Float64Array(6),
      output = new Float64Array(4);
    assert.throws(() => matmul(a, b, output, 2, 2, 4, false, false, 0), /smaller/);
    assert.throws(() => matmul(a, b, output, -1, 2, 3, false, false, 0), /dimensions/);
    assert.throws(
      () => matmul(new Float32Array(6), b, output, 2, 2, 3, false, false, 0),
      /Float64/,
    );
    assert.throws(() => matmul(a, b, a.subarray(1, 5), 2, 2, 3, false, false, 0), /overlap/);
  },
);
