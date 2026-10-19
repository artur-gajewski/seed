import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync, spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ModelStudio } from '../server/studio.ts';

const script = fileURLToPath(new URL('../scripts/train-model.ts', import.meta.url));

test('training CLI saves a new model, resumes it, and rejects invalid settings without replacing it', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'seed-train-cli-'));
  const corpus = path.join(directory, 'corpus.txt');
  const checkpoint = path.join(directory, 'checkpoint.json');
  const run = (...args: string[]) =>
    spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });
  try {
    await writeFile(corpus, 'User: Hello!\nAssistant: Hello there.\n\n'.repeat(8));
    const trained = run(
      '--corpus',
      corpus,
      '--checkpoint',
      checkpoint,
      '--embedding-width',
      '16',
      '--layers',
      '1',
      '--attention-heads',
      '2',
      '--batch-size',
      '1',
      '--steps',
      '2',
      '--learning-rate',
      '0.002',
      '--context-window',
      '8',
    );
    assert.equal(trained.status, 0, trained.stderr);
    assert.match(trained.stdout, /Finished at update 2/);
    const studio = new ModelStudio(checkpoint);
    await studio.loadCheckpoint();
    assert.deepEqual(studio.getStatus().trainingSettings, {
      hiddenSize: 16,
      layers: 1,
      heads: 2,
      batchSize: 1,
      steps: 2,
      learningRate: 0.002,
      sequenceLength: 8,
    });
    const resumed = run(
      '--corpus',
      corpus,
      '--checkpoint',
      checkpoint,
      '--resume',
      '--steps',
      '3',
      '--embedding-width',
      '16',
      '--layers',
      '1',
      '--attention-heads',
      '2',
      '--context-window',
      '8',
    );
    assert.equal(resumed.status, 0, resumed.stderr);
    await studio.loadCheckpoint();
    assert.equal(studio.getStatus().model!.steps, 5);
    assert.equal(studio.getStatus().trainingSettings!.batchSize, 8);
    assert.equal(studio.getStatus().trainingSettings!.learningRate, 0.0003);
    const saved = await readFile(checkpoint);
    for (const args of [
      ['--resume', '--embedding-width', '32'],
      ['--embedding-width', '513'],
      ['--learning-rate', 'oops'],
      ['--context-window', '512'],
    ]) {
      const invalid = run('--corpus', corpus, '--checkpoint', checkpoint, ...args);
      assert.equal(invalid.status, 1);
      assert.deepEqual(await readFile(checkpoint), saved);
    }
    const missingBoth = run();
    assert.equal(missingBoth.status, 1);
    assert.match(missingBoth.stderr, /--corpus PATH \(input file\)/);
    assert.match(missingBoth.stderr, /--checkpoint PATH \(output file\)/);
    const missingOutput = run('--corpus', corpus);
    assert.equal(missingOutput.status, 1);
    assert.match(missingOutput.stderr, /--checkpoint PATH/);
    const missingInput = run('--checkpoint', checkpoint);
    assert.equal(missingInput.status, 1);
    assert.match(missingInput.stderr, /--corpus PATH/);
    assert.equal(run('--help').status, 0);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('training CLI reports remaining time at each 100-update boundary and completion', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'seed-train-progress-'));
  try {
    const corpus = path.join(directory, 'corpus.txt');
    await writeFile(corpus, 'abc'.repeat(100));
    const run = spawnSync(
      process.execPath,
      [
        script,
        '--corpus',
        corpus,
        '--checkpoint',
        path.join(directory, 'checkpoint.json'),
        '--embedding-width',
        '16',
        '--layers',
        '1',
        '--attention-heads',
        '2',
        '--context-window',
        '8',
        '--batch-size',
        '1',
        '--steps',
        '205',
      ],
      { encoding: 'utf8' },
    );
    assert.equal(run.status, 0, run.stderr);
    for (const step of [100, 200, 205]) {
      const lines = run.stdout.split('\n').filter((line) => line.startsWith(`${step}/205`));
      assert.equal(lines.length, 1, run.stdout);
      assert.match(lines[0], /remaining \d{2}(?::\d{2}){0,2}$/);
    }
    assert.match(run.stdout, /205\/205 .*remaining 00/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('training CLI saves progress on SIGINT', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'seed-train-signal-'));
  const corpus = path.join(directory, 'corpus.txt');
  const checkpoint = path.join(directory, 'checkpoint.json');
  try {
    await writeFile(corpus, 'abc'.repeat(100));
    const child = spawn(process.execPath, [
      script,
      '--corpus',
      corpus,
      '--checkpoint',
      checkpoint,
      '--embedding-width',
      '16',
      '--layers',
      '1',
      '--attention-heads',
      '2',
      '--context-window',
      '8',
      '--batch-size',
      '1',
      '--steps',
      '1000000',
    ]);
    let output = '';
    let errors = '';
    let sent = false;
    child.stdout.on('data', (chunk) => {
      output += chunk.toString();
      if (!sent && output.includes('Checkpoint:')) {
        sent = true;
        child.kill('SIGINT');
      }
    });
    child.stderr.on('data', (chunk) => {
      errors += chunk.toString();
    });
    const code = await new Promise<number | null>((resolve, reject) => {
      child.on('error', reject);
      child.on('close', resolve);
    });
    assert.equal(code, 130, errors);
    assert.match(output, /Stopped at update/);
    const studio = new ModelStudio(checkpoint);
    await studio.loadCheckpoint();
    assert.ok(studio.getStatus().model);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
