import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

interface BlasBinding {
  matmul: (
    left: Float64Array,
    right: Float64Array,
    output: Float64Array,
    rows: number,
    columns: number,
    inner: number,
    transposeLeft: boolean,
    transposeRight: boolean,
    beta: number,
  ) => void;
}

const requested = process.env.SEED_TENSOR_BACKEND ?? 'auto';
if (!['auto', 'javascript', 'accelerate', 'openblas'].includes(requested))
  throw new Error('SEED_TENSOR_BACKEND must be auto, javascript, accelerate, or openblas');
const candidate =
  requested === 'auto' ? (process.platform === 'darwin' ? 'accelerate' : 'openblas') : requested;
const supported =
  candidate === 'accelerate'
    ? process.platform === 'darwin'
    : candidate === 'openblas' && process.platform === 'linux';
const addon = new URL(`../native/${candidate}.node`, import.meta.url);
let binding: BlasBinding | null = null;
if (supported && existsSync(addon)) {
  // Surface broken addons instead of silently falling back to slower execution.
  binding = createRequire(import.meta.url)(fileURLToPath(addon)) as BlasBinding;
}
if (requested !== 'auto' && requested !== 'javascript' && !binding)
  throw new Error(
    `${requested} is unavailable. Run npm run native:build on a supported platform, then restart.`,
  );

export const tensorBackend = binding ? candidate : 'javascript';
export const nativeMatmul = binding?.matmul ?? null;
