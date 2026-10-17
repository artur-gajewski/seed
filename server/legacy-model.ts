export const parameterKeys = [
  'embeddings',
  'recurrentWeights',
  'hiddenBias',
  'outputWeights',
  'outputBias',
] as const;

export type ParameterKey = (typeof parameterKeys)[number];
type ParameterTensors = Record<ParameterKey, Float64Array>;

// Keep the original checkpoint keys so existing saved models still load.
const checkpointKeys = {
  embeddings: 'E',
  recurrentWeights: 'W',
  hiddenBias: 'b',
  outputWeights: 'O',
  outputBias: 'c',
} as const;

type CheckpointParameterKey = (typeof checkpointKeys)[ParameterKey];
type SerializedTensors = Record<CheckpointParameterKey, number[]>;

export interface ModelCheckpoint {
  version: 1;
  vocab: string[];
  hiddenSize: number;
  step: number;
  p: SerializedTensors;
  m: SerializedTensors;
  v: SerializedTensors;
}

interface ForwardPass {
  hiddenState: Float64Array;
  probabilities: Float64Array;
}

interface TrainingResult {
  loss: number;
  gradients: ParameterTensors;
}

const MAX_GRADIENT_NORM = 5;
const ADAM_FIRST_MOMENT_DECAY = 0.9;
const ADAM_SECOND_MOMENT_DECAY = 0.999;
const ADAM_EPSILON = 1e-8;

function createZeroTensors(parameters: ParameterTensors): ParameterTensors {
  return {
    embeddings: new Float64Array(parameters.embeddings.length),
    recurrentWeights: new Float64Array(parameters.recurrentWeights.length),
    hiddenBias: new Float64Array(parameters.hiddenBias.length),
    outputWeights: new Float64Array(parameters.outputWeights.length),
    outputBias: new Float64Array(parameters.outputBias.length),
  };
}

function serializeTensors(parameters: ParameterTensors): SerializedTensors {
  return {
    E: Array.from(parameters.embeddings),
    W: Array.from(parameters.recurrentWeights),
    b: Array.from(parameters.hiddenBias),
    O: Array.from(parameters.outputWeights),
    c: Array.from(parameters.outputBias),
  };
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function softmax(logits: Float64Array): Float64Array {
  const probabilities = new Float64Array(logits.length);
  let largestLogit = -Infinity;

  for (const logit of logits) {
    largestLogit = Math.max(largestLogit, logit);
  }

  // Subtracting the largest logit prevents overflow without changing the result.
  let total = 0;

  for (let token = 0; token < logits.length; token++) {
    probabilities[token] = Math.exp(logits[token] - largestLogit);
    total += probabilities[token];
  }

  for (let token = 0; token < probabilities.length; token++) {
    probabilities[token] /= total;
  }

  return probabilities;
}

function sampleToken(probabilities: Float64Array, topK: number): number {
  let candidates = Array.from(probabilities, (probability, token) => ({ probability, token }));
  candidates.sort((left, right) => right.probability - left.probability);

  if (topK > 0) {
    candidates = candidates.slice(0, topK);
  }

  // Sampling uses only the retained probability mass when top K is enabled.
  const totalProbability = candidates.reduce(
    (total, candidate) => total + candidate.probability,
    0,
  );
  let remainingProbability = Math.random() * totalProbability;

  for (const candidate of candidates) {
    remainingProbability -= candidate.probability;

    if (remainingProbability <= 0) {
      return candidate.token;
    }
  }

  return candidates.at(-1)!.token;
}

// A character-level Elman RNN with handwritten backpropagation and Adam.
// Matrices are stored row by row in flat arrays: row * rowWidth + column.
export class LanguageModel {
  readonly vocabulary: string[];
  readonly tokenByCharacter: Map<string, number>;
  readonly hiddenSize: number;
  readonly vocabularySize: number;
  trainingSteps = 0;
  weights: ParameterTensors;
  firstMoments: ParameterTensors;
  secondMoments: ParameterTensors;

  constructor(text: string, hiddenSize = 48, seed = 42) {
    this.vocabulary = [...new Set(Array.from(text))].sort();
    this.tokenByCharacter = new Map(this.vocabulary.map((character, token) => [character, token]));
    this.hiddenSize = hiddenSize;
    this.vocabularySize = this.vocabulary.length;

    const random = createSeededRandom(seed);
    const randomWeights = (size: number, scale: number) =>
      Float64Array.from({ length: size }, () => (random() * 2 - 1) * scale);

    this.weights = {
      embeddings: randomWeights(this.vocabularySize * hiddenSize, 0.08),
      recurrentWeights: randomWeights(hiddenSize * hiddenSize, Math.sqrt(3 / hiddenSize)),
      hiddenBias: new Float64Array(hiddenSize),
      outputWeights: randomWeights(this.vocabularySize * hiddenSize, 0.08),
      outputBias: new Float64Array(this.vocabularySize),
    };
    this.firstMoments = createZeroTensors(this.weights);
    this.secondMoments = createZeroTensors(this.weights);
  }

  encode(text: string): number[] {
    const characters = Array.from(text);
    const unknownCharacters = [
      ...new Set(characters.filter((character) => !this.tokenByCharacter.has(character))),
    ];

    if (unknownCharacters.length > 0) {
      const examples = unknownCharacters.slice(0, 12).map((character) => JSON.stringify(character));
      throw new Error(
        `Characters outside the training vocabulary: ${examples.join(', ')}. Add them to the corpus and train a new model.`,
      );
    }

    return characters.map((character) => this.tokenByCharacter.get(character)!);
  }

  forward(token: number, previousHiddenState: Float64Array, temperature = 1): ForwardPass {
    const hiddenState = this.computeHiddenState(token, previousHiddenState);
    const logits = this.computeOutputLogits(hiddenState, temperature);

    return { hiddenState, probabilities: softmax(logits) };
  }

  private computeHiddenState(token: number, previousHiddenState: Float64Array): Float64Array {
    const { hiddenSize, weights } = this;
    const hiddenState = new Float64Array(hiddenSize);

    // Combine the current character's embedding with the previous recurrent state.
    for (let unit = 0; unit < hiddenSize; unit++) {
      let activation = weights.embeddings[token * hiddenSize + unit] + weights.hiddenBias[unit];

      for (let previousUnit = 0; previousUnit < hiddenSize; previousUnit++) {
        activation +=
          weights.recurrentWeights[unit * hiddenSize + previousUnit] *
          previousHiddenState[previousUnit];
      }

      hiddenState[unit] = Math.tanh(activation);
    }

    return hiddenState;
  }

  private computeOutputLogits(hiddenState: Float64Array, temperature: number): Float64Array {
    const { hiddenSize, vocabularySize, weights } = this;
    const logits = new Float64Array(vocabularySize);

    for (let token = 0; token < vocabularySize; token++) {
      let logit = weights.outputBias[token];

      for (let unit = 0; unit < hiddenSize; unit++) {
        logit += weights.outputWeights[token * hiddenSize + unit] * hiddenState[unit];
      }

      logits[token] = logit / temperature;
    }

    return logits;
  }

  lossAndGradients(
    inputs: number[],
    targets: number[],
    initialHiddenState: Float64Array = new Float64Array(this.hiddenSize),
  ): TrainingResult {
    const gradients = createZeroTensors(this.weights);
    const hiddenStates: Float64Array[] = [initialHiddenState];
    const predictions: Float64Array[] = [];
    let totalLoss = 0;

    // Remember each forward pass so the backward pass can reuse its activations.
    for (let position = 0; position < inputs.length; position++) {
      const { hiddenState, probabilities } = this.forward(inputs[position], hiddenStates[position]);
      hiddenStates.push(hiddenState);
      predictions.push(probabilities);
      totalLoss -= Math.log(Math.max(1e-300, probabilities[targets[position]]));
    }

    let futureHiddenGradient: Float64Array = new Float64Array(this.hiddenSize);

    // Walk backward through the sampled window: each state influences later states.
    for (let position = inputs.length - 1; position >= 0; position--) {
      const hiddenState = hiddenStates[position + 1];
      const previousHiddenState = hiddenStates[position];

      // For softmax cross-entropy, dLoss/dLogit = probability - target indicator.
      const outputGradient = predictions[position].slice();
      outputGradient[targets[position]] -= 1;
      const hiddenGradient = futureHiddenGradient.slice();

      this.accumulateOutputGradients(outputGradient, hiddenState, hiddenGradient, gradients);
      futureHiddenGradient = this.accumulateRecurrentGradients(
        inputs[position],
        hiddenState,
        previousHiddenState,
        hiddenGradient,
        gradients,
      );
    }

    // Average both loss and gradients so the update scale does not depend on window length.
    for (const values of Object.values(gradients)) {
      for (let index = 0; index < values.length; index++) {
        values[index] /= inputs.length;
      }
    }

    return { loss: totalLoss / inputs.length, gradients };
  }

  private accumulateOutputGradients(
    outputGradient: Float64Array,
    hiddenState: Float64Array,
    hiddenGradient: Float64Array,
    gradients: ParameterTensors,
  ): void {
    const { hiddenSize, vocabularySize, weights } = this;

    for (let token = 0; token < vocabularySize; token++) {
      gradients.outputBias[token] += outputGradient[token];

      for (let unit = 0; unit < hiddenSize; unit++) {
        const weightIndex = token * hiddenSize + unit;
        gradients.outputWeights[weightIndex] += outputGradient[token] * hiddenState[unit];
        hiddenGradient[unit] += weights.outputWeights[weightIndex] * outputGradient[token];
      }
    }
  }

  private accumulateRecurrentGradients(
    token: number,
    hiddenState: Float64Array,
    previousHiddenState: Float64Array,
    hiddenGradient: Float64Array,
    gradients: ParameterTensors,
  ): Float64Array {
    const { hiddenSize, weights } = this;
    const previousHiddenGradient = new Float64Array(hiddenSize);

    for (let unit = 0; unit < hiddenSize; unit++) {
      // The derivative of tanh is 1 - tanh(x)^2.
      const activationGradient = hiddenGradient[unit] * (1 - hiddenState[unit] * hiddenState[unit]);
      gradients.hiddenBias[unit] += activationGradient;
      gradients.embeddings[token * hiddenSize + unit] += activationGradient;

      for (let previousUnit = 0; previousUnit < hiddenSize; previousUnit++) {
        const weightIndex = unit * hiddenSize + previousUnit;
        gradients.recurrentWeights[weightIndex] +=
          activationGradient * previousHiddenState[previousUnit];
        previousHiddenGradient[previousUnit] +=
          weights.recurrentWeights[weightIndex] * activationGradient;
      }
    }

    return previousHiddenGradient;
  }

  train(inputs: number[], targets: number[], learningRate: number): number {
    const { loss, gradients } = this.lossAndGradients(inputs, targets);
    const gradientScale = this.computeGradientScale(gradients);
    this.trainingSteps++;
    this.applyAdamUpdate(gradients, gradientScale, learningRate);

    return loss;
  }

  private computeGradientScale(gradients: ParameterTensors): number {
    let squaredNorm = 0;

    for (const values of Object.values(gradients)) {
      for (const gradient of values) {
        squaredNorm += gradient * gradient;
      }
    }

    // Scale the entire gradient together when its norm exceeds the limit.
    return Math.min(1, MAX_GRADIENT_NORM / (Math.sqrt(squaredNorm) + 1e-12));
  }

  private applyAdamUpdate(
    gradients: ParameterTensors,
    gradientScale: number,
    learningRate: number,
  ): void {
    const firstMomentCorrection = 1 - ADAM_FIRST_MOMENT_DECAY ** this.trainingSteps;
    const secondMomentCorrection = 1 - ADAM_SECOND_MOMENT_DECAY ** this.trainingSteps;

    for (const name of parameterKeys) {
      const parameters = this.weights[name];
      const firstMoments = this.firstMoments[name];
      const secondMoments = this.secondMoments[name];

      for (let index = 0; index < parameters.length; index++) {
        const gradient = gradients[name][index] * gradientScale;
        firstMoments[index] = ADAM_FIRST_MOMENT_DECAY * firstMoments[index] + 0.1 * gradient;
        secondMoments[index] =
          ADAM_SECOND_MOMENT_DECAY * secondMoments[index] + 0.001 * gradient * gradient;

        // Correct the initial zero bias in Adam's running averages.
        const correctedFirstMoment = firstMoments[index] / firstMomentCorrection;
        const correctedSecondMoment = secondMoments[index] / secondMomentCorrection;
        parameters[index] -=
          (learningRate * correctedFirstMoment) / (Math.sqrt(correctedSecondMoment) + ADAM_EPSILON);
      }
    }
  }

  generate(prompt: string, count = 200, temperature = 0.8, topK = 0): string {
    const promptTokens = this.encode(prompt);
    let state: ForwardPass = {
      hiddenState: new Float64Array(this.hiddenSize),
      probabilities: new Float64Array(this.vocabularySize),
    };

    // An empty prompt uses the first vocabulary character as an implicit seed.
    for (const token of promptTokens.length > 0 ? promptTokens : [0]) {
      state = this.forward(token, state.hiddenState, temperature);
    }

    let completion = '';

    for (let position = 0; position < count; position++) {
      const token = sampleToken(state.probabilities, topK);
      completion += this.vocabulary[token];
      state = this.forward(token, state.hiddenState, temperature);
    }

    return completion;
  }

  get parameterCount(): number {
    return Object.values(this.weights).reduce((total, values) => total + values.length, 0);
  }

  serialize(): ModelCheckpoint {
    return {
      version: 1,
      vocab: this.vocabulary,
      hiddenSize: this.hiddenSize,
      step: this.trainingSteps,
      p: serializeTensors(this.weights),
      m: serializeTensors(this.firstMoments),
      v: serializeTensors(this.secondMoments),
    };
  }

  static restore(checkpoint: ModelCheckpoint): LanguageModel {
    if (checkpoint.version !== 1) {
      throw new Error('Unsupported checkpoint version');
    }

    const model = new LanguageModel(checkpoint.vocab.join(''), checkpoint.hiddenSize);

    if (JSON.stringify(model.vocabulary) !== JSON.stringify(checkpoint.vocab)) {
      throw new Error('Invalid vocabulary');
    }

    const tensorGroups = [
      { destination: model.weights, saved: checkpoint.p },
      { destination: model.firstMoments, saved: checkpoint.m },
      { destination: model.secondMoments, saved: checkpoint.v },
    ];

    for (const { destination, saved } of tensorGroups) {
      for (const name of parameterKeys) {
        const values = saved?.[checkpointKeys[name]];

        if (
          !Array.isArray(values) ||
          values.length !== destination[name].length ||
          values.some((value) => !Number.isFinite(value))
        ) {
          throw new Error('Invalid checkpoint tensors');
        }

        destination[name] = Float64Array.from(values);
      }
    }

    model.trainingSteps = checkpoint.step;
    return model;
  }
}
