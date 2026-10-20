import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import path from 'node:path';
import type { ChatMessage } from '../shared/api.ts';
import { ModelStudio } from '../server/studio.ts';

type Example = { question: string; answer: string; family: string; history?: ChatMessage[] };

const { values } = parseArgs({
  options: {
    generate: { type: 'boolean' },
    question: { type: 'string' },
    checkpoint: { type: 'string', default: 'data/models/checkpoint.json' },
    output: { type: 'string' },
    evaluation: { type: 'string', default: 'data/references/color_reasoning_01_reference.json' },
    temperature: { type: 'string', default: '0.4' },
    'top-k': { type: 'string', default: '1' },
  },
});
if (values.generate) {
  await import('./create-color-reasoning-01.ts');
} else if (values.question) {
  const studio = new ModelStudio(values.checkpoint);
  await studio.loadCheckpoint();
  console.log(
    studio.chat({
      messages: [{ role: 'user', content: values.question }],
      temperature: Number(values.temperature),
      topK: Number(values['top-k']),
      maxTokens: 48,
    }).reply,
  );
} else {
  const output =
    values.output ??
    path.join(
      'data/reports',
      path.basename(values.evaluation).replace(/(?:_reference)?\.json$/i, '_report.json'),
    );
  if (path.resolve(output) === path.resolve(values.evaluation)) {
    throw new Error('The report output must differ from the reference evaluation dataset');
  }
  const examples = JSON.parse(await readFile(values.evaluation, 'utf8')) as Example[];
  const studio = new ModelStudio(values.checkpoint);
  await studio.loadCheckpoint();
  const results = examples.map((example) => {
    const { reply } = studio.chat({
      messages: [...(example.history ?? []), { role: 'user', content: example.question }],
      temperature: Number(values.temperature),
      topK: Number(values['top-k']),
      maxTokens: 48,
    });
    const passed =
      reply
        .trim()
        .toLowerCase()
        .replace(/[.!?]+$/, '') === example.answer;
    return { ...example, reply, passed };
  });
  const summary = Object.fromEntries(
    [...new Set(results.map((r) => r.family))].map((family) => {
      const group = results.filter((r) => r.family === family);
      return [family, { correct: group.filter((r) => r.passed).length, total: group.length }];
    }),
  );
  const report = {
    checkpoint: values.checkpoint,
    evaluation: values.evaluation,
    steps: studio.getStatus().model?.steps,
    temperature: Number(values.temperature),
    topK: Number(values['top-k']),
    summary,
    correct: results.filter((r) => r.passed).length,
    total: results.length,
    results,
  };
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ ...report, output, results: undefined }, null, 2));
}
