import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { clearLine, cursorTo } from 'node:readline';
import { ModelStudio } from '../server/studio.ts';
import { tensorBackend } from '../server/numerics.ts';
import type { TrainingJob } from '../shared/api.ts';
import { formatDuration } from '../src/lib/formatDuration.ts';

try {
  const { values } = parseArgs({
    options: {
      corpus: { type: 'string' },
      checkpoint: { type: 'string' },
      'embedding-width': { type: 'string' },
      layers: { type: 'string' },
      'attention-heads': { type: 'string' },
      'batch-size': { type: 'string' },
      steps: { type: 'string' },
      'learning-rate': { type: 'string' },
      'context-window': { type: 'string' },
      resume: { type: 'boolean' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.help) {
    console.log(`Train locally without starting the web server.

Usage: npm run train -- --corpus PATH --checkpoint PATH [options]
  --corpus PATH          Required UTF-8 training text file
  --checkpoint PATH      Required output checkpoint; also loaded when using --resume
  --embedding-width N    Default: 256; range: 16–1024, divisible by attention heads
  --layers N             Default: 4; range: 1–12
  --attention-heads N    Default: 8; range: 1–16
  --batch-size N         Default: 8; range: 1–8
  --steps N              Default: 30000; range: 1–1000000 (additional updates on resume)
  --learning-rate N      Default: 0.0003; range: 0.00001–0.05
  --context-window N     Default: 256 byte tokens; range: 8–256
  --resume               Load the checkpoint and continue its learned weights and optimizer

Omitted training settings always use the defaults listed above, including on resume.
Architecture options must match the checkpoint. The learning-rate schedule restarts.
Progress includes estimated time remaining every 100 session updates and every 5 seconds.
Checkpoints replace the selected file every 100 total updates and at completion.
Ctrl+C or SIGTERM stops training and saves progress before exiting.
Use a separate checkpoint path if the web studio is also running.`);
  } else {
    const missing = [
      !values.corpus?.trim() && '--corpus PATH (input file)',
      !values.checkpoint?.trim() && '--checkpoint PATH (output file)',
    ].filter(Boolean);
    if (missing.length)
      throw new Error(`Missing required arguments: ${missing.join(', ')}. Use --help for options.`);
    if (!values.corpus || !values.checkpoint)
      throw new Error('Input and output files are required.');
    const text = await readFile(path.resolve(values.corpus), 'utf8');
    const checkpoint = path.resolve(values.checkpoint);
    const studio = new ModelStudio(checkpoint);
    if (values.resume) await studio.loadCheckpoint();
    const saved = studio.getStatus().trainingSettings;
    const settings = {
      hiddenSize: Number(values['embedding-width'] ?? 256),
      layers: Number(values.layers ?? 4),
      heads: Number(values['attention-heads'] ?? 8),
      batchSize: Number(values['batch-size'] ?? 8),
      steps: Number(values.steps ?? 30_000),
      learningRate: Number(values['learning-rate'] ?? 0.0003),
      sequenceLength: Number(values['context-window'] ?? 256),
    };
    if (values.resume) {
      for (const key of ['hiddenSize', 'layers', 'heads', 'sequenceLength'] as const) {
        if (saved?.[key] !== settings[key]) {
          throw new Error(`Cannot change ${key} on resume; start a new model instead.`);
        }
      }
    }

    let interrupted = false;
    const stop = () => {
      interrupted = true;
      studio.stopTraining();
    };
    let progressLineActive = false;
    const finishProgressLine = () => {
      if (progressLineActive) {
        process.stdout.write('\n');
        progressLineActive = false;
      }
    };
    process.once('SIGINT', stop);
    process.once('SIGTERM', stop);
    try {
      let lastProgress = Date.now();
      let lastReportedUpdate = -1;
      const reportProgress = (job: Readonly<TrainingJob>) => {
        const remaining =
          job.estimatedRemainingMs === undefined
            ? 'calculating'
            : formatDuration(job.estimatedRemainingMs);
        const line = `${job.completed}/${job.total} · loss ${job.loss?.toFixed(4) ?? 'pending'} · validation ${job.validationLoss?.toFixed(4) ?? 'pending'} · remaining ${remaining}`;
        if (process.stdout.isTTY) {
          cursorTo(process.stdout, 0);
          clearLine(process.stdout, 0);
          process.stdout.write(line);
          progressLineActive = true;
        } else {
          console.log(line);
        }
        lastProgress = Date.now();
        lastReportedUpdate = job.completed;
      };
      studio.startTraining({ text, ...settings, resume: values.resume ?? false }, (job) => {
        if (job.completed % 100 === 0) reportProgress(job);
      });
      console.log(`Training ${path.resolve(values.corpus)} · ${tensorBackend}`);
      console.log(`Settings: ${JSON.stringify(settings)}`);
      console.log(`Checkpoint: ${checkpoint}`);
      while (studio.getStatus().running) {
        await new Promise<void>((resolve) => setTimeout(resolve, 200));
        const { job, running } = studio.getStatus();
        if (
          job &&
          job.completed !== lastReportedUpdate &&
          (Date.now() - lastProgress >= 5000 || !running)
        ) {
          reportProgress(job);
        }
      }
      const status = studio.getStatus();
      if (status.error) throw new Error(status.error);
      await studio.saveCheckpoint();
      finishProgressLine();
      console.log(
        `${interrupted ? 'Stopped' : 'Finished'} at update ${status.model!.steps}. Saved ${checkpoint}`,
      );
      if (interrupted) process.exitCode = 130;
    } finally {
      finishProgressLine();
      process.removeListener('SIGINT', stop);
      process.removeListener('SIGTERM', stop);
    }
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
