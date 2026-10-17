import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const directory = new URL('../corpus-examples/', import.meta.url);
const evaluation = JSON.parse(
  await readFile(new URL('toddler_reasoning_evaluation.json', directory), 'utf8'),
) as Array<{ family: string; question: string }>;
const reserved = new Set(evaluation.map(({ family }) => family));
const questions = new Set<string>();
const groups: string[][] = [];
let current: string[];

function add(question: string, answer: string) {
  assert.ok(!questions.has(question), `Duplicate question: ${question}`);
  assert.ok(!evaluation.some((example) => example.question === question));
  const text = `User: ${question}\nAssistant: ${answer}`;
  assert.ok(Buffer.byteLength(text) + 2 <= 256, `Example exceeds context: ${question}`);
  questions.add(question);
  current.push(text);
}

for (const operation of ['add', 'subtract', 'multiply', 'divide'] as const) {
  for (let x = 0; x <= (operation === 'add' || operation === 'subtract' ? 20 : 10); x++) {
    for (let y = 0; y <= (operation === 'add' || operation === 'subtract' ? 20 : 10); y++) {
      if (operation === 'subtract' && y > x) continue;
      if (operation === 'divide' && y === 0) continue;
      const family =
        operation === 'add'
          ? `add:${Math.min(x, y)}:${Math.max(x, y)}:0`
          : `${operation}:${x}:${y}:0`;
      if (reserved.has(family)) continue;
      current = [];
      groups.push(current);
      const left = operation === 'divide' ? x * y : x;
      const result =
        operation === 'add'
          ? x + y
          : operation === 'subtract'
            ? x - y
            : operation === 'multiply'
              ? x * y
              : x;
      const symbol = { add: '+', subtract: '-', multiply: '*', divide: '/' }[operation];
      const word = { add: 'plus', subtract: 'minus', multiply: 'times', divide: 'divided by' }[
        operation
      ];
      const equation = `${left} ${word} ${y} is ${result}.`;
      add(`What is ${left} ${symbol} ${y}?`, equation);
      add(`What is ${left} ${word} ${y}?`, equation);
      add(`Fill in the blank: ${left} ${symbol} ${y} = ?`, `${left} ${symbol} ${y} = ${result}.`);
      if (operation === 'add') {
        add(
          `A box holds ${x} beads. I add ${y} more. How many beads are in the box?`,
          `There are ${result} beads. ${equation}`,
        );
        add(
          `Start at ${x} and count forward ${y} steps. Where do you land?`,
          `You land on ${result}. ${equation}`,
        );
      } else if (operation === 'subtract') {
        add(
          `A box holds ${x} beads. I remove ${y}. How many beads remain?`,
          `There are ${result} beads left. ${equation}`,
        );
        add(
          `Start at ${x} and count backward ${y} steps. Where do you land?`,
          `You land on ${result}. ${equation}`,
        );
      } else if (operation === 'multiply') {
        add(
          `There are ${x} bags with ${y} beads in each bag. How many beads are there in total?`,
          `There are ${result} beads. ${equation}`,
        );
        add(`What is ${x} groups of ${y}?`, equation);
      } else {
        add(
          `Share ${left} beads equally among ${y} cups. How many beads go in each cup?`,
          `Each cup gets ${result} beads. ${equation}`,
        );
        add(
          `How many groups of ${y} can you make from ${left}?`,
          `You can make ${result} groups. ${equation}`,
        );
      }
    }
  }
}

for (let x = 0; x <= 20; x++) {
  current = [];
  groups.push(current);
  add(`What number comes after ${x}?`, `${x + 1} comes after ${x}.`);
  if (x > 0) add(`What number comes before ${x}?`, `${x - 1} comes before ${x}.`);
  add(`Is ${x} even or odd?`, `${x} is ${x % 2 === 0 ? 'even' : 'odd'}.`);
  const end = Math.min(x + 5, 25);
  add(
    `Count from ${x} to ${end}.`,
    `${Array.from({ length: end - x + 1 }, (_, i) => x + i).join(', ')}.`,
  );
  for (let y = 0; y <= 20; y++) {
    add(
      `Is ${x} greater than ${y}?`,
      `${x > y ? 'Yes' : 'No'}. ${x} is ${x > y ? 'greater than' : x === y ? 'equal to' : 'less than'} ${y}.`,
    );
    add(
      `Are ${x} and ${y} equal?`,
      `${x === y ? 'Yes' : 'No'}. ${x} and ${y} are ${x === y ? 'equal' : 'not equal'}.`,
    );
  }
}

// Keep operand families together; distribute topics throughout the file reproducibly.
const hash = (group: string[]) => createHash('sha256').update(group[0]).digest('hex');
groups.sort((a, b) => hash(a).localeCompare(hash(b)));
const examples = groups.flat();
await writeFile(new URL('basic_math.txt', directory), `${examples.join('\n\n')}\n`);
console.log(`Wrote ${examples.length} unique basic math examples, each within 256 bytes.`);
