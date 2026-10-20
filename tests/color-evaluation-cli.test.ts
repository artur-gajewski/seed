import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { ModelStudio } from '../server/studio.ts';

const run = promisify(execFile);
const script = fileURLToPath(new URL('../scripts/evaluate-color-reasoning.ts', import.meta.url));

test('color evaluator defaults to reports by dataset and supports nested output paths', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-color-report-'));
  try {
    const studio = new ModelStudio(path.join(directory, 'data/models/checkpoint.json'));
    studio.startTraining({
      text: 'User: What color is a red boat?\nAssistant: red.\n\n'.repeat(3),
      steps: 1,
      hiddenSize: 16,
      layers: 1,
      heads: 1,
      batchSize: 1,
      sequenceLength: 16,
    });
    const deadline = Date.now() + 5000;
    while (studio.getStatus().running && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.equal(studio.getStatus().running, false);
    assert.equal(studio.getStatus().error, null);
    await studio.saveCheckpoint();
    const dataset = 'data/references/color_reasoning_02_reference.json';
    await mkdir(path.dirname(path.join(directory, dataset)), { recursive: true });
    const source = JSON.stringify([
      { question: 'What color is a red boat?', answer: 'red', family: 'direct' },
    ]);
    await writeFile(path.join(directory, dataset), source);
    const { stdout } = await run(process.execPath, [script, '--evaluation', dataset], {
      cwd: directory,
    });
    const summary = JSON.parse(stdout);
    assert.equal(summary.output, 'data/reports/color_reasoning_02_report.json');
    const report = JSON.parse(await readFile(path.join(directory, summary.output), 'utf8'));
    assert.equal(report.total, 1);
    assert.equal(report.steps, 1);
    assert.equal(report.checkpoint, 'data/models/checkpoint.json');
    const output = 'custom/nested/report.json';
    await run(process.execPath, [script, '--evaluation', dataset, '--output', output], {
      cwd: directory,
    });
    assert.equal(JSON.parse(await readFile(path.join(directory, output), 'utf8')).total, 1);
    assert.equal(await readFile(path.join(directory, dataset), 'utf8'), source);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('color evaluator refuses to overwrite its reference dataset with a report', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-color-reference-'));
  try {
    const dataset = path.join(directory, 'questions.json');
    const source = '[{"question":"A red boat. Color?","answer":"red","family":"direct"}]';
    await writeFile(dataset, source);
    await assert.rejects(
      run(process.execPath, [script, '--evaluation', dataset, '--output', dataset]),
      /report output must differ from the reference evaluation dataset/,
    );
    assert.equal(await readFile(dataset, 'utf8'), source);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
