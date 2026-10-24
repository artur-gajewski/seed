import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { parseCorpus } from './lib/compositional-generalization.ts';

const { values } = parseArgs({
  options: {
    corpus: { type: 'string', default: 'data/inputs/compositional_generalization.txt' },
    output: { type: 'string', default: 'data/reports/compositional_generalization_analysis.json' },
  },
});
if (path.resolve(values.corpus) === path.resolve(values.output)) {
  throw new Error('Analysis output must differ from the corpus');
}
const text = await readFile(values.corpus, 'utf8');
const examples = parseCorpus(text);
const questions = new Map<string, Set<string>>();
const families: Record<string, { total: number; answers: Record<string, number> }> = {};
for (const example of examples) {
  const answers = questions.get(example.question) ?? new Set<string>();
  answers.add(example.answer);
  questions.set(example.question, answers);
  const group = (families[example.family] ??= { total: 0, answers: {} });
  group.total++;
  group.answers[example.answer] = (group.answers[example.answer] ?? 0) + 1;
}
const report = {
  corpus: values.corpus,
  bytes: Buffer.byteLength(text),
  total: examples.length,
  uniqueQuestions: questions.size,
  duplicateQuestions: examples.length - questions.size,
  conflictingAnswers: [...questions]
    .filter(([, answers]) => answers.size > 1)
    .map(([question, answers]) => ({ question, answers: [...answers] })),
  families,
  unclassified: examples.filter((example) => example.family === 'unclassified'),
};
await mkdir(path.dirname(values.output), { recursive: true });
await writeFile(values.output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ ...report, output: values.output }, null, 2));
