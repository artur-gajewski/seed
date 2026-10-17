import { cpus, platform, arch } from 'node:os';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { LanguageModel as TransformerModel } from '../server/model.ts';
import { tensorBackend } from '../server/numerics.ts';

const modelModule = process.env.BENCH_MODEL_PATH
  ? pathToFileURL(resolve(process.env.BENCH_MODEL_PATH)).href
  : new URL('../server/model.ts', import.meta.url).href;
const { LanguageModel } = (await import(modelModule)) as { LanguageModel: typeof TransformerModel };

// Fixed seed, deterministic byte windows, and constant learning rate isolate engine performance.
const samples = Number(process.env.BENCH_SAMPLES ?? 3);
const warmup = Number(process.env.BENCH_WARMUP ?? 1);
const width = Number(process.env.BENCH_WIDTH ?? 256);
const context = Number(process.env.BENCH_CONTEXT ?? 256);
const batchSize = Number(process.env.BENCH_BATCH ?? 8);
const layers = Number(process.env.BENCH_LAYERS ?? 4);
const heads = Number(process.env.BENCH_HEADS ?? 8);
if (
  ![samples, warmup, batchSize].every(Number.isInteger) ||
  samples < 1 ||
  warmup < 0 ||
  batchSize < 1
)
  throw new Error('Invalid benchmark sample, warmup, or batch count');
const model = new LanguageModel('', width, 42, { layers, heads, contextSize: context });
const batch = Array.from({ length: batchSize }, (_, sample) => ({
  inputs: Array.from({ length: context }, (_, position) => (position + sample * 17) % 256),
  targets: Array.from({ length: context }, (_, position) => (position + sample * 17 + 1) % 256),
}));
for (let step = 0; step < warmup; step++) model.trainBatch(batch, 0.0003);
const milliseconds: number[] = [];
const losses: number[] = [];
for (let step = 0; step < samples; step++) {
  const start = performance.now();
  losses.push(model.trainBatch(batch, 0.0003));
  milliseconds.push(performance.now() - start);
  console.error(`Update ${step + 1}/${samples}: ${milliseconds.at(-1)!.toFixed(1)} ms`);
}
const sorted = [...milliseconds].sort((a, b) => a - b);
const median = (sorted[Math.floor((samples - 1) / 2)] + sorted[Math.floor(samples / 2)]) / 2;
console.log(
  JSON.stringify(
    {
      node: process.version,
      backend: tensorBackend,
      modelModule,
      host: `${platform()} ${arch()} ${cpus()[0]?.model}`,
      config: model.config,
      parameters: model.parameterCount,
      batchSize,
      warmup,
      milliseconds,
      medianMs: median,
      hoursFor30000: (median * 30000) / 3600000,
      losses,
    },
    null,
    2,
  ),
);
