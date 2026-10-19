import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { basicMathExamples } from '../scripts/create-basic-math-corpus.ts';

test('basic math corpus fits the context, has correct equations, and excludes held-out families', async () => {
  const directory = new URL('../corpus-examples/', import.meta.url);
  const text = await readFile(new URL('toddler_reasoning_corpus.txt', directory), 'utf8');
  const evaluation = JSON.parse(
    await readFile(new URL('toddler_reasoning_evaluation.json', directory), 'utf8'),
  ) as Array<{ family: string; question: string }>;
  const reserved = new Set(evaluation.map(({ family }) => family));
  const basicQuestions = new Set(basicMathExamples.map((example) => example.split('\n')[0]));
  const examples = text
    .trim()
    .split('\n\n')
    .filter((example) => basicQuestions.has(example.split('\n')[0]));
  const mergedQuestions = new Set([...text.matchAll(/^User: (.+)$/gm)].map((match) => match[1]));
  const questions = new Set<string>();
  let equations = 0;
  for (const example of examples) {
    assert.ok(Buffer.byteLength(example) + 2 <= 256);
    const match = /^User: ([^\n]+)\nAssistant: ([^\n]+)$/.exec(example);
    assert.ok(match);
    assert.ok(!questions.has(match[1]));
    questions.add(match[1]);
    assert.ok(mergedQuestions.has(match[1]), `Missing merged question: ${match[1]}`);
    for (const equation of match[2].matchAll(
      /(\d+) (plus|minus|times|divided by|\+|-|\*|\/) (\d+) (?:is|=) (\d+)/g,
    )) {
      const [, left, operation, right, answer] = equation;
      const x = Number(left);
      const y = Number(right);
      const result = ['plus', '+'].includes(operation)
        ? x + y
        : ['minus', '-'].includes(operation)
          ? x - y
          : ['times', '*'].includes(operation)
            ? x * y
            : x / y;
      assert.equal(Number(answer), result);
      const family = ['plus', '+'].includes(operation)
        ? `add:${Math.min(x, y)}:${Math.max(x, y)}:0`
        : ['minus', '-'].includes(operation)
          ? `subtract:${x}:${y}:0`
          : undefined;
      assert.ok(!family || !reserved.has(family));
      equations++;
    }
  }
  assert.equal(examples.length, 5295);
  assert.equal(equations, 4330);
  for (const { question } of evaluation) assert.ok(!questions.has(question));
});
