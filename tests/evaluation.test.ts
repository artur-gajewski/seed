import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { LanguageModel } from '../server/model.ts';
import {
  evaluateExamples,
  parseEvaluationExamples,
  scoreAnswer,
  summarizeEvaluation,
} from '../server/evaluation.ts';

const story = {
  question: 'Mia has 4 cookies and receives 2 more. How many now?',
  answer: 'Mia has 6 cookies because 4 plus 2 is 6.',
  family: 'add:2:4:0',
};
const combined = {
  question: 'What is 7 − 3 + 2?',
  answer: '7 minus 3 is 4. Then 4 plus 2 is 6.',
  family: 'give-receive:7:3:2',
};

test('evaluation distinguishes numerical answers from name and explanation errors', () => {
  assert.equal(scoreAnswer(story, story.answer).passed, true);
  assert.equal(scoreAnswer(story, '6').passed, true);
  assert.equal(scoreAnswer(story, 'The answer is 6.').passed, true);
  assert.equal(scoreAnswer(story, '6 cookies.').passed, true);
  const wrongName = scoreAnswer(story, 'Eli has 6 cookies because 4 plus 2 is 6.');
  assert.equal(wrongName.numericCorrect, true);
  assert.equal(wrongName.subjectCorrect, false);
  assert.equal(wrongName.passed, false);
  const wrongEquation = scoreAnswer(story, 'Mia has 6 cookies because 4 plus 2 is 8.');
  assert.equal(wrongEquation.numericCorrect, true);
  assert.equal(wrongEquation.equationsCorrect, false);
  assert.equal(wrongEquation.passed, false);
  const unrelated = scoreAnswer(story, 'Mia has 6 cookies because 3 plus 3 is 6.');
  assert.equal(unrelated.equationsCorrect, true);
  assert.equal(unrelated.explanationMatches, false);
  assert.equal(unrelated.passed, false);
  assert.equal(scoreAnswer(story, '1 1 and he pears 1 becar.').predicted, null);
  assert.equal(scoreAnswer(story, 'and gives away 1. How many are left?').predicted, null);
  assert.equal(
    scoreAnswer(story, 'Eli has 8 cookies because 4 plus 2 is 8.').numericCorrect,
    false,
  );
});

test('evaluation recognizes full step-by-step equations without mistaking operands for answers', () => {
  assert.equal(scoreAnswer(combined, combined.answer).predicted, 6);
  assert.equal(scoreAnswer(combined, combined.answer).passed, true);
  assert.equal(scoreAnswer(combined, '7 − 3 = 4. Then 4 + 2 = 6.').passed, true);
  assert.equal(scoreAnswer(story, '2 plus 4 is 6.').passed, true);
  assert.equal(scoreAnswer(combined, '7 minus 3 is 6.').passed, false);
  assert.equal(scoreAnswer(combined, '').predicted, null);
});

test('evaluation validates the actual dataset and rejects malformed references', async () => {
  const source = await readFile(
    new URL('../corpus-examples/toddler_reasoning_evaluation.json', import.meta.url),
    'utf8',
  );
  const examples = parseEvaluationExamples(JSON.parse(source));
  assert.equal(examples.length, 528);
  for (const example of examples) assert.equal(scoreAnswer(example, example.answer).passed, true);
  assert.throws(() => parseEvaluationExamples([]), /nonempty/);
  assert.throws(
    () => parseEvaluationExamples([{ ...story, answer: '4 plus 2 is 8.' }]),
    /reference/,
  );
  assert.throws(
    () => parseEvaluationExamples([{ ...story, answer: '3 plus 3 is 6.' }]),
    /arithmetic family/,
  );
  assert.throws(
    () => parseEvaluationExamples([{ ...story, family: 'unknown' }]),
    /Invalid example/,
  );
});

test('evaluation uses fresh chats and summarizes accuracy, failures, and response times', async () => {
  const replies = [story.answer, '8'];
  const progress: number[] = [];
  let index = 0;
  const results = await evaluateExamples(
    [story, combined],
    (request) => {
      assert.deepEqual(request.messages, [
        { role: 'user', content: [story, combined][index].question },
      ]);
      assert.equal(request.topK, 1);
      const reply = replies[index++];
      return { reply, elapsedMs: index * 10, modelStep: 40000, tokens: 20 };
    },
    { temperature: 0.4, topK: 1, maxTokens: 160 },
    (_, completed) => progress.push(completed),
  );
  assert.deepEqual(progress, [1, 2]);
  const summary = summarizeEvaluation(results);
  assert.equal(summary.numericAccuracy, 0.5);
  assert.equal(summary.passRate, 0.5);
  assert.equal(summary.meanResponseMs, 15);
  assert.equal(summary.medianResponseMs, 15);
  assert.equal(summary.p95ResponseMs, 20);
  assert.equal(summary.byOperation.add.passed, 1);
  assert.equal(summary.byOperation['give-receive'].passed, 0);
  assert.throws(() => summarizeEvaluation([]), /No evaluation/);
});

test('evaluation rejects a model changing during a run', async () => {
  let step = 0;
  await assert.rejects(
    evaluateExamples(
      [story, combined],
      () => ({
        reply: '6',
        elapsedMs: 1,
        modelStep: step++,
        tokens: 1,
      }),
      { temperature: 0.4, topK: 1, maxTokens: 160 },
    ),
    /Model changed/,
  );
});

test('evaluation CLI reads a separate checkpoint and writes a report without modifying it', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-evaluation-'));
  try {
    const checkpoint = path.join(directory, 'checkpoint.json');
    const dataset = path.join(directory, 'questions.json');
    const output = path.join(directory, 'report.json');
    const original = JSON.stringify({
      model: new LanguageModel('', 16, 42, { layers: 1, heads: 2, contextSize: 64 }).serialize(),
    });
    await writeFile(checkpoint, original);
    await writeFile(dataset, JSON.stringify([story, combined]));
    const run = spawnSync(
      process.execPath,
      [
        fileURLToPath(new URL('../scripts/evaluate-model.ts', import.meta.url)),
        '--checkpoint',
        checkpoint,
        '--dataset',
        dataset,
        '--output',
        output,
        '--limit',
        '1',
        '--max-tokens',
        '1',
      ],
      { encoding: 'utf8' },
    );
    assert.equal(run.status, 0, run.stderr);
    const report = JSON.parse(await readFile(output, 'utf8'));
    assert.equal(report.summary.total, 1);
    assert.equal(report.model.steps, 0);
    assert.equal(report.generationSettings.topK, 1);
    assert.equal(report.results[0].question, story.question);
    assert.equal(report.datasetSha256.length, 64);
    assert.deepEqual(
      report.incorrectAnswers,
      report.results.filter((result: { passed: boolean }) => !result.passed),
    );
    assert.equal(await readFile(checkpoint, 'utf8'), original);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
