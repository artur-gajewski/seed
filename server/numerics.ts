import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

interface AccelerateBinding {
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
if (!['auto', 'javascript', 'accelerate'].includes(requested))
  throw new Error('SEED_TENSOR_BACKEND must be auto, javascript, or accelerate');
const addon = new URL('../native/accelerate.node', import.meta.url);
let binding: AccelerateBinding | null = null;
if (requested !== 'javascript' && process.platform === 'darwin' && existsSync(addon)) {
  // A present but broken addon must surface its load error rather than silently losing acceleration.
  binding = createRequire(import.meta.url)(fileURLToPath(addon)) as AccelerateBinding;
}
if (requested === 'accelerate' && !binding)
  throw new Error('Accelerate is unavailable. On macOS run npm run native:build, then restart.');

export const tensorBackend = binding ? 'accelerate' : 'javascript';
export const nativeMatmul = binding?.matmul ?? null;
