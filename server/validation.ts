import type { GenerateRequest, TrainRequest } from '../shared/api.ts';

export type JsonObject = Record<string, unknown>;

function requireInteger(value: unknown, minimum: number, maximum: number, label: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be an integer from ${minimum} to ${maximum}`);
  }

  return value;
}

function requireNumber(value: unknown, minimum: number, maximum: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be from ${minimum} to ${maximum}`);
  }

  return value;
}

export function parseTrainingRequest(body: JsonObject): TrainRequest {
  if (typeof body.text !== 'string' || Array.from(body.text).length < 64) {
    throw new Error('Supply at least 64 characters of training text');
  }

  if (Buffer.byteLength(body.text) > 10_000_000) {
    throw new Error('Corpus exceeds 10 MB');
  }

  const hiddenSize = requireInteger(body.hiddenSize ?? 64, 16, 512, 'Hidden size');
  const steps = requireInteger(body.steps ?? 1000, 1, 1000000, 'Steps');
  const sequenceLength = requireInteger(body.sequenceLength ?? 64, 8, 256, 'Sequence length');
  const learningRate = requireNumber(body.learningRate ?? 0.001, 0.00001, 0.05, 'Learning rate');
  const layers = requireInteger(body.layers ?? 2, 1, 4, 'Layers');
  const heads = requireInteger(body.heads ?? 4, 1, 8, 'Attention heads');
  const batchSize = requireInteger(body.batchSize ?? 2, 1, 8, 'Batch size');
  if (hiddenSize % heads !== 0) {
    throw new Error('Hidden size must be divisible by the number of attention heads');
  }

  return {
    text: body.text,
    hiddenSize,
    steps,
    sequenceLength,
    learningRate,
    resume: Boolean(body.resume),
    layers,
    heads,
    batchSize,
  };
}

export function parseGenerationRequest(body: JsonObject): GenerateRequest {
  if (typeof body.prompt !== 'string' || Array.from(body.prompt).length > 4096) {
    throw new Error('Prompt must be text up to 4096 characters');
  }

  return {
    prompt: body.prompt,
    maxTokens: requireInteger(body.maxTokens ?? 200, 1, 1000, 'Max tokens'),
    temperature: requireNumber(body.temperature ?? 0.8, 0.1, 2, 'Temperature'),
    topK: requireInteger(body.topK ?? 0, 0, 512, 'Top K'),
  };
}

export function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
