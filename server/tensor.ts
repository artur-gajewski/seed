import { nativeMatmul } from './numerics.ts';
import { acceleratedAttention } from './accelerated-attention.ts';

// Small matrix operations and their exact derivatives. Optional native double-precision BLAS.
export interface Tensor {
  rows: number;
  columns: number;
  values: Float64Array;
  gradients: Float64Array;
}

export type Tape = Array<() => void> | null;

export function tensor(rows: number, columns: number, values?: Float64Array): Tensor {
  return {
    rows,
    columns,
    values: values ?? new Float64Array(rows * columns),
    gradients: new Float64Array(rows * columns),
  };
}

export function add(left: Tensor, right: Tensor, tape: Tape): Tensor {
  const output = tensor(left.rows, left.columns);
  for (let index = 0; index < output.values.length; index++)
    output.values[index] = left.values[index] + right.values[index];
  tape?.push(() => {
    for (let index = 0; index < output.values.length; index++) {
      left.gradients[index] += output.gradients[index];
      right.gradients[index] += output.gradients[index];
    }
  });
  return output;
}

export function linear(input: Tensor, weights: Tensor, bias: Tensor, tape: Tape): Tensor {
  const output = tensor(input.rows, weights.columns);
  const rows = input.rows;
  const features = input.columns;
  const columns = weights.columns;
  const x = input.values;
  const w = weights.values;
  const y = output.values;
  for (let row = 0; row < input.rows; row++) {
    y.set(bias.values, row * columns);
  }
  if (nativeMatmul) {
    nativeMatmul(x, w, y, rows, columns, features, false, false, 1);
  } else {
    // Visit contiguous weight/output rows, retaining each output's original summation order.
    for (let row = 0; row < rows; row++) {
      const outputOffset = row * columns;
      for (let feature = 0; feature < features; feature++) {
        const value = x[row * features + feature];
        const weightOffset = feature * columns;
        for (let column = 0; column < columns; column++)
          y[outputOffset + column] += value * w[weightOffset + column];
      }
    }
  }
  tape?.push(() => {
    const dy = output.gradients;
    const dx = input.gradients;
    const dw = weights.gradients;
    for (let row = 0; row < rows; row++)
      for (let column = 0; column < columns; column++)
        bias.gradients[column] += dy[row * columns + column];
    if (nativeMatmul) {
      // += is essential: residual paths and earlier batch windows already contribute gradients.
      nativeMatmul(dy, w, dx, rows, features, columns, false, true, 1);
      nativeMatmul(x, dy, dw, features, columns, rows, true, false, 1);
    } else {
      for (let row = 0; row < rows; row++) {
        const outputOffset = row * columns;
        for (let feature = 0; feature < features; feature++) {
          const inputIndex = row * features + feature;
          const value = x[inputIndex];
          const weightOffset = feature * columns;
          let gradient = dx[inputIndex];
          for (let column = 0; column < columns; column++) {
            const outputGradient = dy[outputOffset + column];
            gradient += w[weightOffset + column] * outputGradient;
            dw[weightOffset + column] += value * outputGradient;
          }
          dx[inputIndex] = gradient;
        }
      }
    }
  });
  return output;
}

export function layerNorm(input: Tensor, scale: Tensor, bias: Tensor, tape: Tape): Tensor {
  const output = tensor(input.rows, input.columns);
  const normalized = new Float64Array(input.values.length);
  const inverseDeviations = new Float64Array(input.rows);
  for (let row = 0; row < input.rows; row++) {
    const start = row * input.columns;
    let mean = 0;
    for (let unit = 0; unit < input.columns; unit++) mean += input.values[start + unit];
    mean /= input.columns;
    let variance = 0;
    for (let unit = 0; unit < input.columns; unit++)
      variance += (input.values[start + unit] - mean) ** 2;
    inverseDeviations[row] = 1 / Math.sqrt(variance / input.columns + 1e-5);
    for (let unit = 0; unit < input.columns; unit++) {
      normalized[start + unit] = (input.values[start + unit] - mean) * inverseDeviations[row];
      output.values[start + unit] =
        normalized[start + unit] * scale.values[unit] + bias.values[unit];
    }
  }
  tape?.push(() => {
    for (let row = 0; row < input.rows; row++) {
      const start = row * input.columns;
      let gradientSum = 0;
      let normalizedGradientSum = 0;
      for (let unit = 0; unit < input.columns; unit++) {
        const gradient = output.gradients[start + unit] * scale.values[unit];
        gradientSum += gradient;
        normalizedGradientSum += gradient * normalized[start + unit];
      }
      for (let unit = 0; unit < input.columns; unit++) {
        const index = start + unit;
        const gradient = output.gradients[index];
        scale.gradients[unit] += gradient * normalized[index];
        bias.gradients[unit] += gradient;
        input.gradients[index] +=
          inverseDeviations[row] *
          (gradient * scale.values[unit] -
            gradientSum / input.columns -
            (normalized[index] * normalizedGradientSum) / input.columns);
      }
    }
  });
  return output;
}

export function gelu(input: Tensor, tape: Tape): Tensor {
  const output = tensor(input.rows, input.columns);
  const factor = Math.sqrt(2 / Math.PI);
  const activations = tape ? new Float64Array(input.values.length) : null;
  for (let index = 0; index < input.values.length; index++) {
    const value = input.values[index];
    const activation = Math.tanh(factor * (value + 0.044715 * (value * value * value)));
    if (activations) activations[index] = activation;
    output.values[index] = 0.5 * value * (1 + activation);
  }
  tape?.push(() => {
    for (let index = 0; index < input.values.length; index++) {
      const value = input.values[index];
      const activation = activations![index];
      const derivative =
        0.5 * (1 + activation) +
        0.5 * value * (1 - activation * activation) * factor * (1 + 3 * 0.044715 * value * value);
      input.gradients[index] += output.gradients[index] * derivative;
    }
  });
  return output;
}

// Q, K, and V occupy consecutive thirds of each input row.
export function causalAttention(queryKeyValue: Tensor, heads: number, tape: Tape): Tensor {
  const length = queryKeyValue.rows;
  const width = queryKeyValue.columns / 3;
  const headWidth = width / heads;
  const scoreScale = 1 / Math.sqrt(headWidth);
  const output = tensor(length, width);
  if (nativeMatmul) return acceleratedAttention(queryKeyValue, heads, tape, output);
  const probabilities = new Float64Array(heads * length * length);

  for (let head = 0; head < heads; head++) {
    for (let position = 0; position < length; position++) {
      const probabilityOffset = (head * length + position) * length;
      let maximum = -Infinity;
      for (let source = 0; source <= position; source++) {
        let score = 0;
        for (let unit = 0; unit < headWidth; unit++) {
          const feature = head * headWidth + unit;
          score +=
            queryKeyValue.values[position * 3 * width + feature] *
            queryKeyValue.values[source * 3 * width + width + feature];
        }
        probabilities[probabilityOffset + source] = score * scoreScale;
        maximum = Math.max(maximum, score * scoreScale);
      }
      let total = 0;
      for (let source = 0; source <= position; source++)
        total += probabilities[probabilityOffset + source] = Math.exp(
          probabilities[probabilityOffset + source] - maximum,
        );
      for (let source = 0; source <= position; source++) {
        const probability = (probabilities[probabilityOffset + source] /= total);
        for (let unit = 0; unit < headWidth; unit++) {
          const feature = head * headWidth + unit;
          output.values[position * width + feature] +=
            probability * queryKeyValue.values[source * 3 * width + 2 * width + feature];
        }
      }
    }
  }

  tape?.push(() => {
    const probabilityGradients = new Float64Array(length);
    for (let head = 0; head < heads; head++) {
      for (let position = 0; position < length; position++) {
        const probabilityOffset = (head * length + position) * length;
        let weightedGradient = 0;
        for (let source = 0; source <= position; source++) {
          let gradient = 0;
          for (let unit = 0; unit < headWidth; unit++) {
            const feature = head * headWidth + unit;
            const outputGradient = output.gradients[position * width + feature];
            gradient +=
              outputGradient * queryKeyValue.values[source * 3 * width + 2 * width + feature];
            queryKeyValue.gradients[source * 3 * width + 2 * width + feature] +=
              probabilities[probabilityOffset + source] * outputGradient;
          }
          probabilityGradients[source] = gradient;
          weightedGradient += gradient * probabilities[probabilityOffset + source];
        }
        for (let source = 0; source <= position; source++) {
          const scoreGradient =
            probabilities[probabilityOffset + source] *
            (probabilityGradients[source] - weightedGradient) *
            scoreScale;
          for (let unit = 0; unit < headWidth; unit++) {
            const feature = head * headWidth + unit;
            const queryIndex = position * 3 * width + feature;
            const keyIndex = source * 3 * width + width + feature;
            queryKeyValue.gradients[queryIndex] += scoreGradient * queryKeyValue.values[keyIndex];
            queryKeyValue.gradients[keyIndex] += scoreGradient * queryKeyValue.values[queryIndex];
          }
        }
      }
    }
  });
  return output;
}

export function crossEntropy(logits: Tensor, targets: number[], gradientScale = 0): number {
  let loss = 0;
  for (let row = 0; row < logits.rows; row++) {
    const offset = row * logits.columns;
    let maximum = -Infinity;
    for (let token = 0; token < logits.columns; token++)
      maximum = Math.max(maximum, logits.values[offset + token]);
    let total = 0;
    for (let token = 0; token < logits.columns; token++)
      total += Math.exp(logits.values[offset + token] - maximum);
    loss += Math.log(total) + maximum - logits.values[offset + targets[row]];
    if (gradientScale) {
      for (let token = 0; token < logits.columns; token++)
        logits.gradients[offset + token] +=
          (gradientScale *
            (Math.exp(logits.values[offset + token] - maximum) / total -
              (token === targets[row] ? 1 : 0))) /
          logits.rows;
    }
  }
  return loss / logits.rows;
}
