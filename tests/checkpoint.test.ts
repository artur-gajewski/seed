import test from 'node:test';
import assert from 'node:assert/strict';
import { encodeCheckpoint, decodeCheckpoint } from '../server/checkpoint.ts';

test('binary checkpoints preserve numerical state without creating a JSON string', () => {
  const state = {
    model: { values: [0.12345678901234567, -1e-100], firstMoments: [1e-200, 0] },
    history: [{ step: 100, loss: 1.25 }],
  };
  const encoded = encodeCheckpoint(state);
  assert.equal(encoded[0], 0xff);
  assert.deepEqual(decodeCheckpoint(encoded), state);
  assert.deepEqual(decodeCheckpoint(Buffer.from(JSON.stringify(state))), state);
  assert.throws(() => decodeCheckpoint(encoded.subarray(0, 5)));
  assert.throws(() => decodeCheckpoint(Buffer.from('{broken')));
});
