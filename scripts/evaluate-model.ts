import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash, randomUUID } from 'node:crypto';
import { ModelStudio } from '../server/studio.ts';
import {
  evaluateExamples,
  parseEvaluationExamples,
  summarizeEvaluation,
} from '../server/evaluation.ts';
import { tensorBackend } from '../server/numerics.ts';

const root = fileURLToPath(new URL('../', import.meta.url));
const { values } = parseArgs({
  options: {
    checkpoint: { type: 'string' },
    dataset: { type: 'string' },
    output: { type: 'string' },
    limit: { type: 'string' },
    temperature: { type: 'string' },
    'top-k': { type: 'string' },
    'max-tokens': { type: 'string' },
    help: { type: 'boolean', short: 'h' },
  },
});

if (values.help) {
  console.log(`Evaluate a saved checkpoint without changing the running studio.

Usage: npm run evaluate -- [options]
  --checkpoint PATH   Saved checkpoint (default: data/checkpoint.json)
  --dataset PATH      Evaluation JSON (default: corpus-examples/toddler_reasoning_evaluation.json)
  --limit N           Evenly spaced subset; omit to run all questions
  --temperature N     Default: 0.4
  --top-k N           Default: 1 (greedy, reproducible); 0 enables unrestricted sampling
  --max-tokens N      Default: 160 byte tokens per answer
  --output PATH       JSON report (default: unique file in data/evaluations/)

Each question starts a fresh chat. Reports include all replies, incorrect answers,
numerical accuracy, consistency checks, model settings, and response times.
The checkpoint is read once. No training, saving, or API restart is performed.`);
} else {
  try {
    const checkpoint = path.resolve(values.checkpoint ?? path.join(root, 'data/checkpoint.json'));
    const dataset = path.resolve(
      values.dataset ?? path.join(root, 'corpus-examples/toddler_reasoning_evaluation.json'),
    );
    const source = await readFile(dataset, 'utf8');
    const examples = parseEvaluationExamples(JSON.parse(source));
    const limit = Number(values.limit ?? examples.length);
    const settings = {
      temperature: Number(values.temperature ?? 0.4),
      topK: Number(values['top-k'] ?? 1),
      maxTokens: Number(values['max-tokens'] ?? 160),
    };
    if (!Number.isInteger(limit) || limit < 1 || limit > examples.length)
      throw new Error(`Limit must be between 1 and ${examples.length}`);
    if (
      !Number.isFinite(settings.temperature) ||
      settings.temperature < 0.1 ||
      settings.temperature > 2
    )
      throw new Error('Temperature must be between 0.1 and 2');
    if (!Number.isInteger(settings.topK) || settings.topK < 0 || settings.topK > 512)
      throw new Error('Top K must be an integer between 0 and 512');
    if (
      !Number.isInteger(settings.maxTokens) ||
      settings.maxTokens < 1 ||
      settings.maxTokens > 1000
    )
      throw new Error('Max tokens must be an integer between 1 and 1000');
    const selected = Array.from(
      { length: limit },
      (_, index) => examples[Math.floor((index * examples.length) / limit)],
    );
    console.log(`Loading ${checkpoint}`);
    const studio = new ModelStudio(checkpoint);
    await studio.loadCheckpoint();
    const status = studio.getStatus();
    const startedAt = new Date().toISOString();
    console.log(
      `Step ${status.model!.steps} · ${tensorBackend} · ${selected.length} questions · top K ${settings.topK}`,
    );
    const results = await evaluateExamples(
      selected,
      (request) => studio.chat({ ...request }),
      settings,
      (result, completed) => {
        if (completed % 10 === 0 || completed === selected.length)
          console.log(`${completed}/${selected.length} · last response ${result.elapsedMs} ms`);
      },
    );
    const summary = summarizeEvaluation(results);
    const output = path.resolve(
      values.output ??
        path.join(root, 'data/evaluations', `step-${status.model!.steps}-${randomUUID()}.json`),
    );
    const report = {
      version: 1,
      startedAt,
      finishedAt: new Date().toISOString(),
      checkpoint,
      dataset,
      datasetSha256: createHash('sha256').update(source).digest('hex'),
      model: status.model,
      trainingSettings: status.trainingSettings,
      backend: tensorBackend,
      generationSettings: settings,
      selection: { method: 'evenly-spaced', limit, available: examples.length },
      summary,
      results,
      incorrectAnswers: results.filter((result) => !result.passed),
      scoringNote:
        'Rule-based scoring for this arithmetic dataset. Unrecognized answers count as incorrect. A pass means a correct extracted number with no detected name or equation mismatch; it does not prove reasoning. Review raw replies.',
    };
    await mkdir(path.dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
    console.log(
      `Numerical accuracy: ${summary.numericCorrect}/${summary.total} (${(summary.numericAccuracy * 100).toFixed(1)}%)`,
    );
    console.log(
      `Pass with consistency checks: ${summary.passed}/${summary.total} (${(summary.passRate * 100).toFixed(1)}%)`,
    );
    console.log(`Unrecognized answers: ${summary.unscored} (counted as incorrect)`);
    console.log(
      `Response time: mean ${summary.meanResponseMs.toFixed(0)} ms · median ${summary.medianResponseMs} ms · p95 ${summary.p95ResponseMs} ms`,
    );
    console.log(`Report with incorrect answers: ${output}`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
