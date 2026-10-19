import { add, causalAttention, crossEntropy, gelu, layerNorm, linear, tensor } from './tensor.ts';
import type { Tape, Tensor } from './tensor.ts';
import { LanguageModel as LegacyLanguageModel } from './legacy-model.ts';
import type { ModelCheckpoint as LegacyCheckpoint } from './legacy-model.ts';

export interface TransformerConfig {
  hiddenSize: number;
  layers: number;
  heads: number;
  contextSize: number;
}

interface SerializedParameter {
  rows: number;
  columns: number;
  values: number[];
  firstMoments: number[];
  secondMoments: number[];
}

export interface TransformerCheckpoint {
  version: 2;
  architecture: 'transformer';
  tokenizer: 'utf8-bytes';
  config: TransformerConfig;
  step: number;
  parameters: Record<string, SerializedParameter>;
}

export type ModelCheckpoint = TransformerCheckpoint | LegacyCheckpoint;
export type TrainableModel = LanguageModel | LegacyLanguageModel;
interface Parameter extends Tensor {
  firstMoments: Float64Array;
  secondMoments: Float64Array;
}

const VOCABULARY_SIZE = 258;
const BEGINNING_OF_TEXT = 256;
const END_OF_RESPONSE = 257;

// Decoder-only, pre-normalized Transformer. Every weight starts randomly initialized.
export class LanguageModel {
  readonly config: TransformerConfig;
  readonly weights: Record<string, Parameter> = {};
  readonly vocabularySize = VOCABULARY_SIZE;
  trainingSteps = 0;

  constructor(_text: string, hiddenSize = 64, seed = 42, options: Partial<TransformerConfig> = {}) {
    this.config = { hiddenSize, layers: 2, heads: 4, contextSize: 64, ...options };
    const { layers, heads, contextSize } = this.config;
    if (
      !Number.isInteger(this.hiddenSize) ||
      this.hiddenSize < 4 ||
      this.hiddenSize > 512 ||
      !Number.isInteger(heads) ||
      heads < 1 ||
      heads > 8 ||
      this.hiddenSize % heads !== 0 ||
      !Number.isInteger(layers) ||
      layers < 1 ||
      layers > 4 ||
      !Number.isInteger(contextSize) ||
      contextSize < 2 ||
      contextSize > 256
    ) {
      throw new Error(
        'Invalid Transformer configuration: width must be divisible by heads; maximum width 512, context 256, 4 layers, 8 heads',
      );
    }
    let randomState = seed >>> 0;
    const random = () => {
      randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
      return randomState / 4294967296;
    };
    const parameter = (
      name: string,
      rows: number,
      columns: number,
      initialization: 'random' | 'zero' | 'one' = 'random',
    ) => {
      const values = Float64Array.from({ length: rows * columns }, () =>
        initialization === 'random' ? (random() * 2 - 1) * 0.035 : initialization === 'one' ? 1 : 0,
      );
      this.weights[name] = {
        ...tensor(rows, columns, values),
        firstMoments: new Float64Array(values.length),
        secondMoments: new Float64Array(values.length),
      };
    };
    const width = this.hiddenSize;
    parameter('tokenEmbedding', VOCABULARY_SIZE, width);
    parameter('positionEmbedding', contextSize, width);
    for (let layer = 0; layer < layers; layer++) {
      const prefix = `block${layer}`;
      for (const normalization of ['attentionNorm', 'feedForwardNorm']) {
        parameter(`${prefix}.${normalization}.scale`, 1, width, 'one');
        parameter(`${prefix}.${normalization}.bias`, 1, width, 'zero');
      }
      parameter(`${prefix}.queryKeyValue`, width, 3 * width);
      parameter(`${prefix}.queryKeyValueBias`, 1, 3 * width, 'zero');
      parameter(`${prefix}.attentionProjection`, width, width);
      parameter(`${prefix}.attentionProjectionBias`, 1, width, 'zero');
      parameter(`${prefix}.feedForwardExpansion`, width, 4 * width);
      parameter(`${prefix}.feedForwardExpansionBias`, 1, 4 * width, 'zero');
      parameter(`${prefix}.feedForwardProjection`, 4 * width, width);
      parameter(`${prefix}.feedForwardProjectionBias`, 1, width, 'zero');
    }
    parameter('finalNorm.scale', 1, width, 'one');
    parameter('finalNorm.bias', 1, width, 'zero');
    parameter('outputProjection', width, VOCABULARY_SIZE);
    parameter('outputBias', 1, VOCABULARY_SIZE, 'zero');
  }

  get hiddenSize(): number {
    return this.config.hiddenSize;
  }
  get contextSize(): number {
    return this.config.contextSize;
  }
  get parameterCount(): number {
    return Object.values(this.weights).reduce(
      (count, parameter) => count + parameter.values.length,
      0,
    );
  }

  encode(text: string): number[] {
    return Array.from(Buffer.from(text, 'utf8'));
  }

  encodeTrainingText(text: string): number[] {
    const chunks: number[][] = [[BEGINNING_OF_TEXT]];
    let cursor = 0;
    // Add a learned stop marker after each assistant answer in the dialogue format.
    // Ordinary prose is otherwise kept byte-for-byte, without role conversion.
    for (const match of text.matchAll(/(?:^|\n)Assistant:[\s\S]*?(?=\n\nUser:|$)/g)) {
      const answerEnd = match.index + match[0].trimEnd().length;
      chunks.push(this.encode(text.slice(cursor, answerEnd)), [END_OF_RESPONSE]);
      cursor = answerEnd;
    }
    chunks.push(this.encode(text.slice(cursor)));
    return chunks.flat();
  }

  forward(tokens: number[], tape: Tape = null): Tensor {
    if (
      !tokens.length ||
      tokens.length > this.contextSize ||
      tokens.some((token) => !Number.isInteger(token) || token < 0 || token >= VOCABULARY_SIZE)
    ) {
      throw new Error('Input must contain valid tokens within the model context window');
    }
    let hidden = this.embed(tokens, tape);
    for (let layer = 0; layer < this.config.layers; layer++) {
      const prefix = `block${layer}`;
      const normalized = layerNorm(
        hidden,
        this.weights[`${prefix}.attentionNorm.scale`],
        this.weights[`${prefix}.attentionNorm.bias`],
        tape,
      );
      const queryKeyValue = linear(
        normalized,
        this.weights[`${prefix}.queryKeyValue`],
        this.weights[`${prefix}.queryKeyValueBias`],
        tape,
      );
      const attended = causalAttention(queryKeyValue, this.config.heads, tape);
      const attentionOutput = linear(
        attended,
        this.weights[`${prefix}.attentionProjection`],
        this.weights[`${prefix}.attentionProjectionBias`],
        tape,
      );
      hidden = add(hidden, attentionOutput, tape);
      const feedForwardInput = layerNorm(
        hidden,
        this.weights[`${prefix}.feedForwardNorm.scale`],
        this.weights[`${prefix}.feedForwardNorm.bias`],
        tape,
      );
      const expanded = linear(
        feedForwardInput,
        this.weights[`${prefix}.feedForwardExpansion`],
        this.weights[`${prefix}.feedForwardExpansionBias`],
        tape,
      );
      const activated = gelu(expanded, tape);
      const feedForwardOutput = linear(
        activated,
        this.weights[`${prefix}.feedForwardProjection`],
        this.weights[`${prefix}.feedForwardProjectionBias`],
        tape,
      );
      hidden = add(hidden, feedForwardOutput, tape);
    }
    hidden = layerNorm(
      hidden,
      this.weights['finalNorm.scale'],
      this.weights['finalNorm.bias'],
      tape,
    );
    return linear(hidden, this.weights.outputProjection, this.weights.outputBias, tape);
  }

  private embed(tokens: number[], tape: Tape): Tensor {
    const output = tensor(tokens.length, this.hiddenSize);
    const tokenEmbedding = this.weights.tokenEmbedding;
    const positionEmbedding = this.weights.positionEmbedding;
    for (let position = 0; position < tokens.length; position++) {
      for (let unit = 0; unit < this.hiddenSize; unit++) {
        output.values[position * this.hiddenSize + unit] =
          tokenEmbedding.values[tokens[position] * this.hiddenSize + unit] +
          positionEmbedding.values[position * this.hiddenSize + unit];
      }
    }
    tape?.push(() => {
      for (let position = 0; position < tokens.length; position++) {
        for (let unit = 0; unit < this.hiddenSize; unit++) {
          const gradient = output.gradients[position * this.hiddenSize + unit];
          tokenEmbedding.gradients[tokens[position] * this.hiddenSize + unit] += gradient;
          positionEmbedding.gradients[position * this.hiddenSize + unit] += gradient;
        }
      }
    });
    return output;
  }

  evaluate(inputs: number[], targets: number[]): number {
    this.validateTargets(inputs, targets);
    return crossEntropy(this.forward(inputs), targets);
  }

  private validateTargets(inputs: number[], targets: number[]): void {
    if (
      inputs.length !== targets.length ||
      targets.some((token) => !Number.isInteger(token) || token < 0 || token >= VOCABULARY_SIZE)
    ) {
      throw new Error('Every input token must have a valid next-token target');
    }
  }

  lossAndGradients(
    inputs: number[],
    targets: number[],
  ): { loss: number; gradients: Record<string, Float64Array> } {
    for (const parameter of Object.values(this.weights)) parameter.gradients.fill(0);
    const loss = this.accumulateGradients(inputs, targets, 1);
    return {
      loss,
      gradients: Object.fromEntries(
        Object.entries(this.weights).map(([name, parameter]) => [
          name,
          parameter.gradients.slice(),
        ]),
      ),
    };
  }

  private accumulateGradients(inputs: number[], targets: number[], gradientScale: number): number {
    this.validateTargets(inputs, targets);
    const tape: NonNullable<Tape> = [];
    const loss = crossEntropy(this.forward(inputs, tape), targets, gradientScale);
    for (let operation = tape.length - 1; operation >= 0; operation--) tape[operation]();
    return loss;
  }

  train(inputs: number[], targets: number[], learningRate: number): number {
    return this.trainBatch([{ inputs, targets }], learningRate);
  }

  trainBatch(batch: Array<{ inputs: number[]; targets: number[] }>, learningRate: number): number {
    if (!batch.length) throw new Error('Training batch cannot be empty');
    for (const parameter of Object.values(this.weights)) parameter.gradients.fill(0);
    let loss = 0;
    for (const sample of batch)
      loss += this.accumulateGradients(sample.inputs, sample.targets, 1 / batch.length);
    loss /= batch.length;
    if (!Number.isFinite(loss))
      throw new Error('Training loss is not finite. Reduce the learning rate.');
    this.applyAdamW(learningRate);
    return loss;
  }

  private applyAdamW(learningRate: number): void {
    let squaredNorm = 0;
    for (const parameter of Object.values(this.weights))
      for (const gradient of parameter.gradients) squaredNorm += gradient * gradient;
    if (!Number.isFinite(squaredNorm))
      throw new Error('Training gradients are not finite. Reduce the learning rate.');
    const clippingScale = Math.min(1, 1 / (Math.sqrt(squaredNorm) + 1e-12));
    this.trainingSteps++;
    const firstCorrection = 1 - 0.9 ** this.trainingSteps;
    const secondCorrection = 1 - 0.999 ** this.trainingSteps;
    for (const parameter of Object.values(this.weights)) {
      for (let index = 0; index < parameter.values.length; index++) {
        const gradient = parameter.gradients[index] * clippingScale;
        parameter.firstMoments[index] = 0.9 * parameter.firstMoments[index] + 0.1 * gradient;
        parameter.secondMoments[index] =
          0.999 * parameter.secondMoments[index] + 0.001 * gradient * gradient;
        const correctedMean = parameter.firstMoments[index] / firstCorrection;
        const correctedVariance = parameter.secondMoments[index] / secondCorrection;
        // Decoupled weight decay applies to matrices, not biases or norm scales.
        const decay = parameter.rows > 1 ? 0.01 * parameter.values[index] : 0;
        parameter.values[index] -=
          learningRate * (correctedMean / (Math.sqrt(correctedVariance) + 1e-8) + decay);
      }
    }
  }

  generate(prompt: string, count = 200, temperature = 0.8, topK = 0): string {
    return this.generateDetailed(prompt, count, temperature, topK).completion;
  }

  generateDetailed(
    prompt: string,
    count = 200,
    temperature = 0.8,
    topK = 0,
  ): { completion: string; tokens: number } {
    const context = [BEGINNING_OF_TEXT, ...this.encode(prompt)];
    const generated: number[] = [];
    let generatedTokens = 0;
    for (let position = 0; position < count; position++) {
      const logits = this.forward(context.slice(-this.contextSize));
      const offset = (logits.rows - 1) * VOCABULARY_SIZE;
      // Do not emit the beginning-of-text marker as ordinary text.
      let candidates = Array.from({ length: 257 }, (_, index) => {
        const token = index === 256 ? END_OF_RESPONSE : index;
        return { token, logit: logits.values[offset + token] / temperature };
      });
      candidates.sort((left, right) => right.logit - left.logit);
      if (topK > 0) candidates = candidates.slice(0, topK);
      const maximum = candidates[0].logit;
      const masses = candidates.map((candidate) => Math.exp(candidate.logit - maximum));
      let sample = Math.random() * masses.reduce((total, mass) => total + mass, 0);
      let token = candidates.at(-1)!.token;
      for (let index = 0; index < candidates.length; index++) {
        sample -= masses[index];
        if (sample <= 0) {
          token = candidates[index].token;
          break;
        }
      }
      generatedTokens++;
      if (token === END_OF_RESPONSE) break;
      generated.push(token);
      context.push(token);
    }
    // Incomplete or malformed UTF-8 from an untrained model decodes as replacement characters.
    return { completion: Buffer.from(generated).toString('utf8'), tokens: generatedTokens };
  }

  serialize(): TransformerCheckpoint {
    return {
      version: 2,
      architecture: 'transformer',
      tokenizer: 'utf8-bytes',
      config: { ...this.config },
      step: this.trainingSteps,
      parameters: Object.fromEntries(
        Object.entries(this.weights).map(([name, parameter]) => [
          name,
          {
            rows: parameter.rows,
            columns: parameter.columns,
            values: Array.from(parameter.values),
            firstMoments: Array.from(parameter.firstMoments),
            secondMoments: Array.from(parameter.secondMoments),
          },
        ]),
      ),
    };
  }

  static restore(checkpoint: ModelCheckpoint): TrainableModel {
    if (checkpoint.version === 1) return LegacyLanguageModel.restore(checkpoint);
    if (
      checkpoint.version !== 2 ||
      checkpoint.architecture !== 'transformer' ||
      checkpoint.tokenizer !== 'utf8-bytes' ||
      !Number.isInteger(checkpoint.step) ||
      checkpoint.step < 0
    )
      throw new Error('Unsupported or invalid checkpoint');
    const model = new LanguageModel('', checkpoint.config.hiddenSize, 42, checkpoint.config);
    for (const [name, parameter] of Object.entries(model.weights)) {
      const saved = checkpoint.parameters[name];
      if (!saved || saved.rows !== parameter.rows || saved.columns !== parameter.columns)
        throw new Error(`Invalid checkpoint parameter: ${name}`);
      for (const field of ['values', 'firstMoments', 'secondMoments'] as const) {
        if (
          !Array.isArray(saved[field]) ||
          saved[field].length !== parameter.values.length ||
          saved[field].some((value) => !Number.isFinite(value))
        )
          throw new Error(`Invalid checkpoint values: ${name}`);
        parameter[field].set(saved[field]);
      }
    }
    model.trainingSteps = checkpoint.step;
    return model;
  }
}
