import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('color reasoning evaluation holds complete color/object pairs out of training', async () => {
  const corpus = await readFile(
    new URL('../corpus-examples/color_reasoning.txt', import.meta.url),
    'utf8',
  );
  const examples = JSON.parse(
    await readFile(
      new URL('../corpus-examples/color_reasoning_evaluation.json', import.meta.url),
      'utf8',
    ),
  ) as { question: string; answer: string; family: string }[];
  assert.ok(examples.length > 2000);
  assert.equal(new Set(examples.map((e) => JSON.stringify(e))).size, examples.length);
  for (const example of examples) {
    assert.ok(!corpus.includes(`User: ${example.question}\n`), example.question);
    if (example.family !== 'direct') continue;
    const pair = example.question.match(/(?:a|the) (\w+) (.+)\?$/)!;
    assert.ok(pair);
    assert.equal(example.answer, pair[1]);
    assert.ok(
      !new RegExp(`${pair[1]} ${pair[2]}\\b`).test(corpus),
      `Leaked pair: ${pair[1]} ${pair[2]}`,
    );
    assert.ok(corpus.includes(`Assistant: ${pair[1]}.`));
    assert.ok(corpus.includes(` ${pair[2]}?`));
  }
  assert.ok(examples.some((e) => e.question === 'What color is a red car?'));
  assert.ok(examples.some((e) => e.family === 'conversation'));
  assert.ok(examples.some((e) => e.family === 'unseen-direct'));
  assert.ok(corpus.includes('yellow submarine'));
  assert.ok(corpus.includes('What color is yellow submarine?'));
  for (const object of ['zeppelin', 'ukulele', 'axolotl', 'quadcopter', 'snowmobile', 'teapot']) {
    assert.ok(!corpus.includes(object), `Leaked unseen object: ${object}`);
  }
  assert.ok(!corpus.includes('Assistant: I do not know.'));
  assert.match(corpus, /Assistant: [^\n]+\.\nUser:/);
});
