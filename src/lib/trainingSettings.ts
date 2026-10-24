import type { StudioStatus, TrainingSettings } from '../../shared/api.ts';

export const DEFAULT_TRAINING_SETTINGS: TrainingSettings = {
  hiddenSize: 1024,
  steps: 1_000_000,
  sequenceLength: 2048,
  learningRate: 0.05,
  layers: 12,
  heads: 16,
  batchSize: 8,
};

export function trainingSettingsForModel(status: StudioStatus): TrainingSettings | null {
  const model = status.model;
  if (!model) return null;
  const settings = { ...DEFAULT_TRAINING_SETTINGS, ...status.trainingSettings };
  return {
    ...settings,
    hiddenSize: model.hiddenSize,
    layers: model.layers ?? settings.layers,
    heads: model.heads ?? settings.heads,
    sequenceLength:
      model.contextSize === undefined
        ? settings.sequenceLength
        : Math.min(status.trainingSettings?.sequenceLength ?? model.contextSize, model.contextSize),
  };
}
