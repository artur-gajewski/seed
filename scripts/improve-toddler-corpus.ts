import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { basicMathExamples } from './create-basic-math-corpus.ts';

interface Example {
  question: string;
  answer: string;
  family?: string;
}

const directory = new URL('../corpus-examples/', import.meta.url);
const source = await readFile(new URL('toddler_reasoning_corpus.txt', directory), 'utf8');
const hash = (value: string) => createHash('sha256').update(value).digest().readUInt32LE(0);
const names = ['Mia', 'Ben', 'Anna', 'Eli', 'Ruby', 'Kai', 'Noah', 'Lily'];
const objects = ['cookies', 'blocks', 'apples', 'pencils', 'marbles', 'stickers', 'shells', 'toys'];
const gerunds: Record<string, string> = {
  brush: 'brushing',
  find: 'finding',
  get: 'getting',
  open: 'opening',
  pick: 'picking',
  plant: 'planting',
  pour: 'pouring',
  put: 'putting',
  look: 'looking',
  zip: 'zipping',
  water: 'watering',
  wash: 'washing',
  take: 'taking',
  eat: 'eating',
  walk: 'walking',
  drink: 'drinking',
  read: 'reading',
};

function fixGrammar(text: string): string {
  return text
    .replace(/\ba orange\b/g, 'an orange')
    .replace(/Can (\w+) do /g, 'Can $1 ')
    .replace(/\b(before|after) (\w+)\b/g, (match, preposition, verb) =>
      gerunds[verb] ? `${preposition} ${gerunds[verb]}` : match,
    )
    .replace(/^(\w+)(.*?) comes later\./, (match, verb: string, rest: string) => {
      const gerund = gerunds[verb.toLowerCase()];
      return gerund ? `${gerund[0].toUpperCase()}${gerund.slice(1)}${rest} comes later.` : match;
    })
    .replace(/your teeth/g, 'teeth');
}

const general = new Map<string, Example>();
for (const match of source.matchAll(/User: (.*?)\nAssistant: (.*?)(?=\n\nUser:|$)/gs)) {
  let question = match[1].replace(/^(?:At [^,]+|In [^,]+|On a [^,]+), \w+ asks: /, '');
  let answer = match[2].trim();
  // Rebuild arithmetic while preserving the general examples in the current corpus.
  // Exclude every wording variant of held-out number/operation families.
  if (/\b(?:plus|minus)\b/.test(answer)) continue;
  const comparison = question.match(
    /^(\w+) has a longer (\w+) than (\w+)\. \3 has a longer \2 than (\w+)\./,
  );
  if (comparison) {
    const [, first, object, second, third] = comparison;
    answer = `${first} has the longest ${object} because ${first}'s ${object} is longer than ${second}'s, and ${second}'s is longer than ${third}'s.`;
  }
  question = fixGrammar(question);
  answer = fixGrammar(answer);
  general.set(`${question}\n${answer}`, { question, answer });
}

type Operation = 'add' | 'subtract' | 'give-receive' | 'receive-give';
interface Problem {
  operation: Operation;
  x: number;
  y: number;
  z: number;
  family: string;
}

// These include the questions used during the original investigation. Never train
// on any wording/name/object variant of their arithmetic families.
const reserved = new Set(['add:2:4:0', 'subtract:7:3:0', 'give-receive:7:3:2']);
const isEvaluation = (family: string) => reserved.has(family) || hash(family) % 10 === 0;

function render(problem: Problem, variant: number): Example {
  const { operation, x, y, z, family } = problem;
  const name = names[variant % names.length];
  const other = names[(variant + 1) % names.length];
  const third = names[(variant + 2) % names.length];
  const object = objects[Math.floor(variant / names.length) % objects.length];
  const wording = variant % 4;
  let question: string;
  let answer: string;
  if (operation === 'add') {
    const result = x + y;
    question = [
      `What is ${x} + ${y}?`,
      `${name} has ${x} ${object} and receives ${y} more. How many does ${name} have now?`,
      `${name} has ${x} ${object}. ${other} gives ${name} ${y} more. How many does ${name} have?`,
      `${name} has ${x} ${object} and gets ${y} more. How many ${object} now?`,
    ][wording];
    answer =
      wording === 0
        ? `${x} plus ${y} is ${result}.`
        : `${name} has ${result} ${object} because ${x} plus ${y} is ${result}.`;
  } else if (operation === 'subtract') {
    const result = x - y;
    question = [
      `What is ${x} ${variant % 8 === 0 ? '-' : '−'} ${y}?`,
      `${name} has ${x} ${object} and gives away ${y}. How many remain?`,
      `${name} has ${x} ${object}. ${name} gives ${y} to ${other}. How many does ${name} have left?`,
      `${name} has ${x} ${object} and loses ${y}. How many ${object} are left?`,
    ][wording];
    answer =
      wording === 0
        ? `${x} minus ${y} is ${result}.`
        : `${name} has ${result} ${object} left because ${x} minus ${y} is ${result}.`;
  } else {
    const giveFirst = operation === 'give-receive';
    const intermediate = giveFirst ? x - y : x + y;
    const result = giveFirst ? intermediate + z : intermediate - z;
    const firstAction = giveFirst ? `gives ${y} to ${other}` : `receives ${y} from ${other}`;
    const nextAction = giveFirst ? `receives ${z} from ${third}` : `gives ${z} to ${third}`;
    const firstOperator = giveFirst ? 'minus' : 'plus';
    const nextOperator = giveFirst ? 'plus' : 'minus';
    question =
      wording === 0
        ? `What is ${x} ${giveFirst ? (variant % 8 === 0 ? '-' : '−') : '+'} ${y} ${giveFirst ? '+' : '-'} ${z}?`
        : `${name} has ${x} ${object}. ${name} ${firstAction} and then ${nextAction}. How many does ${name} have?`;
    answer =
      wording === 0
        ? `${x} ${firstOperator} ${y} is ${intermediate}. Then ${intermediate} ${nextOperator} ${z} is ${result}.`
        : `${name} has ${result} ${object}. First, ${x} ${firstOperator} ${y} is ${intermediate}. Then ${intermediate} ${nextOperator} ${z} is ${result}.`;
  }
  return { question, answer, family };
}

const operations: Operation[] = ['add', 'subtract', 'give-receive', 'receive-give'];
const problems = operations.map((operation) => {
  const result: Problem[] = [];
  for (let x = 0; x <= 12; x++) {
    for (let y = 0; y <= 12; y++) {
      for (let z = 0; z <= (operation === 'add' || operation === 'subtract' ? 0 : 8); z++) {
        if ((operation === 'subtract' || operation === 'give-receive') && y > x) continue;
        if (operation === 'receive-give' && z > x + y) continue;
        const family =
          operation === 'add'
            ? `add:${Math.min(x, y)}:${Math.max(x, y)}:0`
            : `${operation}:${x}:${y}:${z}`;
        result.push({ operation, x, y, z, family });
      }
    }
  }
  return result.sort((a, b) => hash(a.family) - hash(b.family));
});

const evaluation = problems.flatMap((group) =>
  group
    .filter((problem) => isEvaluation(problem.family))
    .flatMap((problem) => [render(problem, 0), render(problem, 1)]),
);
const arithmetic = problems.flatMap((group) => {
  const examples = new Map<string, Example>();
  for (let variant = 0; variant < 128 && examples.size < 4000; variant++) {
    for (const problem of group) {
      if (isEvaluation(problem.family)) continue;
      const example = render(problem, variant);
      examples.set(example.question, example);
      if (examples.size === 4000) break;
    }
  }
  if (examples.size !== 4000) throw new Error('Insufficient distinct arithmetic questions');
  return [...examples.values()];
});

const generalExamples = [...general.values()].sort((a, b) => hash(a.question) - hash(b.question));
if (generalExamples.length < 14000) throw new Error('Insufficient general examples');
// Keep arithmetic families together so the studio's contiguous validation split
// does not scatter their wording variants across both partitions.
const merged = new Map<string, Example>();
for (const example of [...generalExamples, ...arithmetic]) merged.set(example.question, example);
for (const text of basicMathExamples) {
  const match = /^User: ([^\n]+)\nAssistant: ([^\n]+)$/.exec(text)!;
  if (!merged.has(match[1])) merged.set(match[1], { question: match[1], answer: match[2] });
}
const training = [...merged.values()].sort(
  (a, b) =>
    hash(a.family ?? a.question) - hash(b.family ?? b.question) ||
    a.question.localeCompare(b.question, 'en'),
);
const evaluationFamilies = new Set(evaluation.map((example) => example.family));
for (const example of training) {
  if (example.family && evaluationFamilies.has(example.family))
    throw new Error('Evaluation leakage');
}
for (const example of [...training, ...evaluation]) {
  const bytes = Buffer.byteLength(`User: ${example.question}\nAssistant: ${example.answer}`) + 2;
  if (bytes > 256) throw new Error(`Example exceeds context: ${bytes} bytes`);
}
const format = (examples: Example[]) =>
  examples.map(({ question, answer }) => `User: ${question}\nAssistant: ${answer}\n`).join('\n');
await writeFile(new URL('toddler_reasoning_corpus.txt', directory), format(training));
await writeFile(new URL('toddler_reasoning_evaluation.txt', directory), format(evaluation));
await writeFile(
  new URL('toddler_reasoning_evaluation.json', directory),
  `${JSON.stringify(evaluation, null, 2)}\n`,
);
console.log(
  `Wrote ${training.length} training examples and ${evaluation.length} evaluation examples.`,
);
