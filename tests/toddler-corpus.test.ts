import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

interface Example {
  question: string;
  answer: string;
}

function parse(text: string): Example[] {
  const examples = [...text.matchAll(/User: ([^\n]+)\nAssistant: ([^\n]+)(?:\n|$)/g)];
  assert.equal(examples.map((match) => match[0].trim()).join('\n\n'), text.trim());
  return examples.map((match) => ({ question: match[1], answer: match[2] }));
}

function arithmeticFamily(answer: string): string | undefined {
  const equations = [...answer.matchAll(/(\d+) (plus|minus) (\d+) is (\d+)/g)];
  if (!equations.length) return undefined;
  for (const [, left, operation, right, result] of equations) {
    assert.equal(
      Number(result),
      operation === 'plus' ? Number(left) + Number(right) : Number(left) - Number(right),
    );
  }
  const [, left, operation, right, result] = equations[0];
  if (equations.length === 1) {
    const x = Number(left);
    const y = Number(right);
    return operation === 'plus'
      ? `add:${Math.min(x, y)}:${Math.max(x, y)}:0`
      : `subtract:${x}:${y}:0`;
  }
  assert.equal(equations.length, 2);
  assert.equal(equations[1][1], result, 'Second operation must use the first result');
  assert.notEqual(equations[1][2], operation);
  return `${operation === 'minus' ? 'give-receive' : 'receive-give'}:${left}:${right}:${equations[1][3]}`;
}

test('toddler corpus has valid arithmetic, bounded examples, and disjoint evaluation families', async () => {
  const directory = new URL('../corpus-examples/', import.meta.url);
  const training = parse(
    await readFile(new URL('toddler_reasoning_corpus.txt', directory), 'utf8'),
  );
  const evaluation = parse(
    await readFile(new URL('toddler_reasoning_evaluation.txt', directory), 'utf8'),
  );
  const manifest = JSON.parse(
    await readFile(new URL('toddler_reasoning_evaluation.json', directory), 'utf8'),
  ) as Array<Example & { family: string }>;
  assert.equal(training.length, 35072);
  assert.equal(new Set(training.map((example) => example.question)).size, 35072);
  assert.deepEqual(
    manifest.map(({ question, answer }) => ({ question, answer })),
    evaluation,
  );
  const trainingFamilies = new Set<string>();
  let arithmeticCount = 0;
  for (const example of [...training, ...evaluation]) {
    assert.ok(
      Buffer.byteLength(`User: ${example.question}\nAssistant: ${example.answer}`) + 2 <= 256,
    );
  }
  for (const example of training) {
    const family = arithmeticFamily(example.answer);
    if (family) {
      trainingFamilies.add(family);
      arithmeticCount++;
      const equations = [...example.answer.matchAll(/(\d+) (plus|minus) (\d+) is (\d+)/g)];
      const inputs = [equations[0][1], equations[0][3]];
      if (equations.length === 2) inputs.push(equations[1][3]);
      assert.deepEqual(example.question.match(/\d+/g), inputs);
      if (/^\w+ has /.test(example.question)) {
        assert.equal(example.answer.split(' ')[0], example.question.split(' ')[0]);
        assert.equal(example.answer.match(/\d+/)![0], equations.at(-1)![4]);
      }
    }
    assert.doesNotMatch(example.answer, /because \w+ is longer than/);
    assert.doesNotMatch(example.question, /Can \w+ do (?:zip|look|put|read|water|eat|drink|walk)/);
  }
  assert.ok(arithmeticCount >= 16000);
  for (const example of manifest) {
    assert.equal(arithmeticFamily(example.answer), example.family);
    assert.equal(trainingFamilies.has(example.family), false);
  }
  for (const family of ['add:2:4:0', 'subtract:7:3:0', 'give-receive:7:3:2']) {
    assert.ok(manifest.some((example) => example.family === family));
    assert.equal(trainingFamilies.has(family), false);
  }
});
