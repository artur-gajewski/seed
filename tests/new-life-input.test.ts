import { NEW_LIFE_PRESETS, newLifeByteLimit } from '../shared/api.ts';
import type { NewLifeSize } from '../shared/api.ts';
import test from 'node:test';
import assert from 'node:assert/strict';
import { newLifeInputContent, parseNewLifeInput } from '../src/lib/newLifeInput.ts';

test('New life composer routes ! answers to the latest question without the command prefix', () => {
  assert.deepEqual(parseNewLifeInput('  Hello!  '), { kind: 'message', message: 'Hello!' });
  assert.deepEqual(parseNewLifeInput(' ! The sky is blue. ', 'What color is the sky?'), {
    kind: 'teach',
    message: 'What color is the sky?',
    answer: 'The sky is blue.',
  });
  assert.throws(() => parseNewLifeInput('!Hello'), /Send a message first/);
  for (const draft of ['', '!', '!   ', '!🙂'.repeat(53), 'a'.repeat(209)]) {
    assert.throws(() => parseNewLifeInput(draft, 'Hi'));
  }
  assert.equal(parseNewLifeInput(`!${'a'.repeat(100)}`, 'Hi').kind, 'teach');
  assert.equal(parseNewLifeInput(`!${'🙂'.repeat(25)}`, 'Hi').kind, 'teach');
});

test('natural correction formats teach the supplied answer rather than the feedback prefix', () => {
  for (const prefix of [
    'No. ',
    'No, ',
    'NO! ',
    'No: ',
    'No — ',
    'Nope, ',
    'Nah, ',
    'Not quite. ',
    "That's wrong. ",
    'That is incorrect: ',
    'That’s not correct. ',
    "You're wrong, ",
    'Wrong. ',
    'Incorrect: ',
    'Actually, ',
    'Correction: ',
    'Instead, ',
    'The correct answer is ',
    'The right answer: ',
    'Correct answer: ',
    'The answer should be ',
    'It should be ',
    'You should say ',
    'Instead say ',
    'I meant ',
    'No, the correct answer is ',
    "No, that's wrong. The correct answer is ",
  ]) {
    const draft = `${prefix}The sky is blue.`;
    assert.deepEqual(
      parseNewLifeInput(draft, 'What color is the sky?'),
      {
        kind: 'teach',
        message: 'What color is the sky?',
        answer: 'The sky is blue.',
      },
      draft,
    );
    assert.deepEqual(newLifeInputContent(draft), {
      isTeaching: true,
      content: 'The sky is blue.',
    });
    assert.throws(() => parseNewLifeInput(draft), /Send a message first/);
  }
  assert.equal(parseNewLifeInput('No, 4.', 'What is 2 + 2?').answer, '4.');
  assert.equal(parseNewLifeInput(`No, ${'a'.repeat(100)}`, 'Hi').kind, 'teach');
  assert.throws(() => parseNewLifeInput(`No, ${'🙂'.repeat(53)}`, 'Hi'));
});

test('ordinary questions, negative statements, and bare feedback do not become taught answers', () => {
  for (const draft of [
    'No.',
    'No,',
    'Actually,',
    'Not quite.',
    'No thanks',
    'No animals live here.',
    'Nobody knows.',
    'What is the correct answer?',
    'Actually, what color is the sky?',
    'No, can you try again?',
    'No, why',
    'The sky is blue.',
    'Is it wrong?',
    'No, blue?',
    'Correction is useful.',
    "No, that's wrong.",
    'No, try again.',
  ]) {
    assert.deepEqual(parseNewLifeInput(draft, 'Hi'), { kind: 'message', message: draft });
  }
  // The explicit command still supports an answer that happens to be a question.
  assert.deepEqual(parseNewLifeInput('!How are you?', 'Hello'), {
    kind: 'teach',
    message: 'Hello',
    answer: 'How are you?',
  });
});

test('New life input limits follow the selected context and leave room for a complete lesson', () => {
  for (const [size, preset] of Object.entries(NEW_LIFE_PRESETS)) {
    const limit = newLifeByteLimit(size as NewLifeSize);
    assert.equal(
      parseNewLifeInput('a'.repeat(limit), undefined, size as NewLifeSize).kind,
      'message',
    );
    assert.equal(
      parseNewLifeInput('!' + 'a'.repeat(limit), 'Hi', size as NewLifeSize).kind,
      'teach',
    );
    assert.throws(() => parseNewLifeInput('a'.repeat(limit + 1), undefined, size as NewLifeSize));
    assert.throws(() =>
      parseNewLifeInput('🙂'.repeat(Math.floor(limit / 4) + 1), undefined, size as NewLifeSize),
    );
    const message = 'a'.repeat(limit);
    const answerLimit = newLifeByteLimit(size as NewLifeSize, message);
    assert.equal(
      parseNewLifeInput('!' + 'b'.repeat(answerLimit), message, size as NewLifeSize).kind,
      'teach',
    );
    assert.throws(() =>
      parseNewLifeInput('!' + 'b'.repeat(answerLimit + 1), message, size as NewLifeSize),
    );
    const lesson = `User: ${message}\nAssistant: ${'b'.repeat(answerLimit)}`;
    assert.ok(Buffer.byteLength(lesson) + 2 <= preset.contextSize);
  }
});

test('153M allows 2000-byte messages and budgets answers from the remaining context', () => {
  assert.equal(newLifeByteLimit('x-large'), 2000);
  assert.equal(newLifeByteLimit('x-large', 'a'.repeat(2000)), 28);
  assert.equal(newLifeByteLimit('x-large', '🙂'.repeat(500)), 28);
});
