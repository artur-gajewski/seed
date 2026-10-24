import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { parseCorpus } from '../scripts/lib/compositional-generalization.ts';

test('compositional generalization parser accepts CRLF and normalizes evaluation answers', () => {
  assert.deepEqual(parseCorpus('User: Who has the cup?\r\nAssistant: Zoe.\r\n'), [
    { question: 'Who has the cup?', answer: 'zoe', family: 'ownership' },
  ]);
  assert.throws(() => parseCorpus('User: Missing answer'), /block 1/);
  assert.throws(() => parseCorpus(''), /block 1/);
});

test('compositional generalization corpus has classified examples and consistent answers', async () => {
  const examples = parseCorpus(
    await readFile(
      new URL('../data/inputs/compositional_generalization.txt', import.meta.url),
      'utf8',
    ),
  );
  assert.ok(examples.length > 0);
  const answers = new Map<string, string>();
  for (const example of examples) {
    assert.notEqual(example.family, 'unclassified', example.question);
    if (answers.has(example.question)) {
      assert.equal(example.answer, answers.get(example.question), example.question);
    }
    answers.set(example.question, example.answer);
  }
});
