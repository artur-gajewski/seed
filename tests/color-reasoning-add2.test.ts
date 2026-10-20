import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { LanguageModel } from '../server/model.ts';

const root = new URL('../data/', import.meta.url);
const read = (name: string) => readFile(new URL(name, root), 'utf8');
interface Example {
  question: string;
  answer: string;
  family: string;
  history?: { role: string; content: string }[];
}

test('color supplement preserves old holdouts and separates its new evaluation pairs', async () => {
  const corpus = await read('inputs/color_reasoning_02.txt');
  const base = await read('inputs/color_reasoning_01.txt');
  const oldEvaluation = JSON.parse(
    await read('references/color_reasoning_01_reference.json'),
  ) as Example[];
  const evaluation = JSON.parse(
    await read('references/color_reasoning_02_reference.json'),
  ) as Example[];
  assert.ok(evaluation.length > 1500);
  assert.equal(new Set(evaluation.map((e) => JSON.stringify(e))).size, evaluation.length);
  for (const example of [...oldEvaluation, ...evaluation]) {
    assert.ok(!corpus.includes(`User: ${example.question}\n`), example.question);
    if (example.family !== 'direct') continue;
    const pair = example.question.match(/(?:a|the) (.+)\?$/)![1];
    assert.ok(!new RegExp(`\\b${pair}\\b`).test(corpus), `Leaked pair: ${pair}`);
  }
  for (const example of evaluation.filter((e) => e.family === 'direct')) {
    const pair = example.question.match(/a (.+)\?$/)![1];
    assert.ok(!base.includes(pair), `New evaluation pair was in earlier training: ${pair}`);
  }
  for (const object of ['zeppelin', 'ukulele', 'axolotl', 'quadcopter', 'snowmobile', 'teapot']) {
    assert.ok(!corpus.includes(object), `Old unseen object leaked: ${object}`);
  }
  assert.ok(evaluation.some((e) => e.family === 'unseen-wording'));
  assert.ok(evaluation.some((e) => e.family === 'long-conversation' && e.history?.length === 6));
});

test('supplement adds accurate color lessons and longer chats within the existing context', async () => {
  const corpus = await read('inputs/color_reasoning_02.txt');
  const blocks = corpus.trim().split(/\n{3,}/);
  assert.ok(blocks.length > 30000);
  assert.equal(new Set(blocks).size, blocks.length);
  const model = new LanguageModel('', 16, 42, { layers: 1, heads: 1, contextSize: 256 });
  let fiveTurns = 0;
  let updates = 0;
  for (const block of blocks) {
    assert.ok(model.encodeTrainingText(block).length <= 256, block);
    const lines = block.split('\n').filter(Boolean);
    assert.equal(lines.length % 2, 0);
    for (let i = 0; i < lines.length; i += 2) {
      assert.match(lines[i], /^User: .+/);
      assert.match(lines[i + 1], /^Assistant: .+\.$/);
      const direct = lines[i].match(/^User: What color is (?:a |the )?(\w+) .+/);
      if (direct) assert.equal(lines[i + 1], `Assistant: ${direct[1]}.`);
    }
    if (lines.length === 10) fiveTurns++;
    const update = block.match(/Paint it (\w+)\. Its color now\?\nAssistant: (\w+)\./);
    if (update) {
      assert.equal(update[1], update[2]);
      assert.ok(block.endsWith(`Assistant: ${update[1]}.`));
      updates++;
    }
  }
  assert.ok(fiveTurns > 500);
  assert.ok(updates > 1000);
  for (const color of [
    'red',
    'blue',
    'green',
    'yellow',
    'black',
    'white',
    'orange',
    'purple',
    'pink',
    'brown',
    'gray',
    'grey',
    'cyan',
    'magenta',
    'silver',
    'gold',
    'turquoise',
    'navy',
    'violet',
    'beige',
    'teal',
    'maroon',
  ]) {
    assert.ok(corpus.includes(`Assistant: ${color}.`), `Missing color: ${color}`);
  }
});
