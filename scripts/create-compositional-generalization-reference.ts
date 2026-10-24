import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { parseCorpus, type CompositionalExample } from './lib/compositional-generalization.ts';

const { values } = parseArgs({
  options: {
    corpus: { type: 'string', default: 'data/inputs/compositional_generalization.txt' },
    output: {
      type: 'string',
      default: 'data/references/compositional_generalization_reference.json',
    },
    'per-family': { type: 'string', default: '100' },
  },
});
const limit = Number(values['per-family']);
if (!Number.isSafeInteger(limit) || limit < 1) {
  throw new Error('--per-family must be a positive integer');
}
if (path.resolve(values.corpus) === path.resolve(values.output)) {
  throw new Error('Reference output must differ from the corpus');
}
const examples = parseCorpus(await readFile(values.corpus, 'utf8'));
const unique = new Map<string, CompositionalExample>();
for (const example of examples) {
  const previous = unique.get(example.question);
  if (previous && previous.answer !== example.answer) {
    throw new Error(`Conflicting answers for: ${example.question}`);
  }
  unique.set(example.question, example);
}
const counts: Record<string, number> = {};
const references = [...unique.values()].filter((example) => {
  const count = counts[example.family] ?? 0;
  if (count >= limit) return false;
  counts[example.family] = count + 1;
  return true;
});
await mkdir(path.dirname(values.output), { recursive: true });
await writeFile(values.output, JSON.stringify(references, null, 2) + '\n');
console.log(JSON.stringify({ output: values.output, total: references.length, families: counts }));
console.log(
  'These questions occur in the corpus; scores measure training-data recall, not holdout accuracy.',
);
