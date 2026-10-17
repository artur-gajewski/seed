import test from 'node:test';
import assert from 'node:assert/strict';
import { parseNewLifeInput } from '../src/lib/newLifeInput.ts';

test('New life composer routes ! answers to the latest question without the command prefix', () => {
  assert.deepEqual(parseNewLifeInput('  Hello!  '), { kind: 'message', message: 'Hello!' });
  assert.deepEqual(parseNewLifeInput(' ! The sky is blue. ', 'What color is the sky?'), {
    kind: 'teach',
    message: 'What color is the sky?',
    answer: 'The sky is blue.',
  });
  assert.throws(() => parseNewLifeInput('!Hello'), /Send a message first/);
  for (const draft of ['', '!', '!   ', '!🙂'.repeat(26), 'a'.repeat(101)]) {
    assert.throws(() => parseNewLifeInput(draft, 'Hi'));
  }
  assert.equal(parseNewLifeInput(`!${'a'.repeat(100)}`, 'Hi').kind, 'teach');
  assert.equal(parseNewLifeInput(`!${'🙂'.repeat(25)}`, 'Hi').kind, 'teach');
});
