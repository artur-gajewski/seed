import test from 'node:test';
import assert from 'node:assert/strict';
import { TrainingTiming } from '../server/training-timing.ts';
import { formatDuration } from '../src/lib/formatDuration.ts';

test('remaining time uses only the last ten updates and waits for ten samples', () => {
  const timing = new TrainingTiming(0);
  for (let update = 1; update < 10; update++) {
    assert.equal(timing.recordUpdate(100 - update, update * 100), undefined);
  }
  assert.equal(timing.recordUpdate(90, 1000), 9000);
  assert.equal(timing.recordUpdate(89, 1200), 9790);
  assert.equal(timing.recordUpdate(0, 1300), 0);
  assert.equal(new TrainingTiming(0).recordUpdate(90, 100), undefined);
});

test('remaining time formats seconds, minutes, and hours at their boundaries', () => {
  for (const [milliseconds, expected] of [
    [0, '00'],
    [1, '01'],
    [59000, '59'],
    [60000, '01:00'],
    [3599000, '59:59'],
    [3600000, '01:00:00'],
    [3661000, '01:01:01'],
    [360000000, '100:00:00'],
  ] as const) {
    assert.equal(formatDuration(milliseconds), expected);
  }
});
