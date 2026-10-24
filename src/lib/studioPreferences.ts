import type { GenerateRequest, TrainingSettings, TrainRequest } from '../../shared/api.ts';

export const DEFAULT_GENERATION_SETTINGS: GenerateRequest = {
  prompt: 'User: Hello!\nAssistant:',
  temperature: 2,
  maxTokens: 1000,
  topK: 512,
};

const numberInRange = (value: unknown, min: number, max: number, integer = true) =>
  typeof value === 'number' &&
  Number.isFinite(value) &&
  (!integer || Number.isInteger(value)) &&
  value >= min &&
  value <= max;

export function isTrainingSettings(value: unknown): value is TrainingSettings {
  if (!value || typeof value !== 'object') return false;
  const s = value as TrainingSettings;
  return (
    numberInRange(s.hiddenSize, 16, 1024) &&
    numberInRange(s.layers, 1, 12) &&
    numberInRange(s.heads, 1, 16) &&
    s.hiddenSize % s.heads === 0 &&
    numberInRange(s.batchSize, 1, 8) &&
    numberInRange(s.sequenceLength, 8, 2048) &&
    numberInRange(s.steps, 1, 1_000_000) &&
    numberInRange(s.learningRate, 0.00001, 0.05, false)
  );
}

export function isGenerationSettings(value: unknown): value is GenerateRequest {
  if (!value || typeof value !== 'object') return false;
  const s = value as GenerateRequest;
  return (
    typeof s.prompt === 'string' &&
    Array.from(s.prompt).length <= 4096 &&
    numberInRange(s.temperature, 0.1, 2, false) &&
    numberInRange(s.maxTokens, 1, 1000) &&
    numberInRange(s.topK, 0, 512)
  );
}

export function isTrainingDevice(value: unknown): value is NonNullable<TrainRequest['device']> {
  return value === 'auto' || value === 'cpu' || value === 'mps' || value === 'cuda';
}

export function readPreference<T>(
  key: string,
  fallback: T,
  valid: (value: unknown) => value is T,
): T {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(key) ?? 'null');
    return valid(saved) ? saved : fallback;
  } catch {
    return fallback;
  }
}

export function writePreference<T>(key: string, value: T, valid: (value: unknown) => value is T) {
  if (!valid(value)) return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Settings remain usable when browser storage is unavailable or full.
  }
}
