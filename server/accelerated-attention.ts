import { nativeMatmul } from './numerics.ts';
import type { Tape, Tensor } from './tensor.ts';

// Pack each head once: BLAS then handles score, value, and gradient matrix products.
// Causal masking and softmax stay identical to the scalar implementation.
export function acceleratedAttention(
  qkv: Tensor,
  heads: number,
  tape: Tape,
  output: Tensor,
): Tensor {
  const matmul = nativeMatmul!;
  const length = qkv.rows;
  const width = output.columns;
  const depth = width / heads;
  const scale = 1 / Math.sqrt(depth);
  for (let head = 0; head < heads; head++) {
    const query = new Float64Array(length * depth);
    const key = new Float64Array(length * depth);
    const value = new Float64Array(length * depth);
    for (let row = 0; row < length; row++)
      for (let unit = 0; unit < depth; unit++) {
        const source = row * 3 * width + head * depth + unit;
        const packed = row * depth + unit;
        query[packed] = qkv.values[source];
        key[packed] = qkv.values[source + width];
        value[packed] = qkv.values[source + 2 * width];
      }
    const probabilities = new Float64Array(length * length);
    matmul(query, key, probabilities, length, length, depth, false, true, 0);
    for (let row = 0; row < length; row++) {
      const offset = row * length;
      let maximum = -Infinity;
      for (let source = 0; source <= row; source++) {
        probabilities[offset + source] *= scale;
        maximum = Math.max(maximum, probabilities[offset + source]);
      }
      let total = 0;
      for (let source = 0; source <= row; source++)
        total += probabilities[offset + source] = Math.exp(
          probabilities[offset + source] - maximum,
        );
      for (let source = 0; source <= row; source++) probabilities[offset + source] /= total;
      probabilities.fill(0, offset + row + 1, offset + length);
    }
    const attended = new Float64Array(length * depth);
    matmul(probabilities, value, attended, length, depth, length, false, false, 0);
    for (let row = 0; row < length; row++)
      output.values.set(
        attended.subarray(row * depth, (row + 1) * depth),
        row * width + head * depth,
      );

    tape?.push(() => {
      const gradient = new Float64Array(length * depth);
      for (let row = 0; row < length; row++)
        for (let unit = 0; unit < depth; unit++)
          gradient[row * depth + unit] = output.gradients[row * width + head * depth + unit];
      const valueGradient = new Float64Array(length * depth);
      matmul(probabilities, gradient, valueGradient, length, depth, length, true, false, 0);
      const scoreGradient = new Float64Array(length * length);
      matmul(gradient, value, scoreGradient, length, length, depth, false, true, 0);
      for (let row = 0; row < length; row++) {
        const offset = row * length;
        let weightedGradient = 0;
        for (let source = 0; source <= row; source++)
          weightedGradient += scoreGradient[offset + source] * probabilities[offset + source];
        for (let source = 0; source <= row; source++)
          scoreGradient[offset + source] =
            probabilities[offset + source] *
            (scoreGradient[offset + source] - weightedGradient) *
            scale;
        scoreGradient.fill(0, offset + row + 1, offset + length);
      }
      const queryGradient = new Float64Array(length * depth);
      const keyGradient = new Float64Array(length * depth);
      matmul(scoreGradient, key, queryGradient, length, depth, length, false, false, 0);
      matmul(scoreGradient, query, keyGradient, length, depth, length, true, false, 0);
      for (let row = 0; row < length; row++)
        for (let unit = 0; unit < depth; unit++) {
          const destination = row * 3 * width + head * depth + unit;
          const packed = row * depth + unit;
          qkv.gradients[destination] += queryGradient[packed];
          qkv.gradients[destination + width] += keyGradient[packed];
          qkv.gradients[destination + 2 * width] += valueGradient[packed];
        }
    });
  }
  return output;
}
