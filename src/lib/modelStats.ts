import type { StudioStatus, TrainingSettings } from '../../shared/api.ts';

export function selectedModelStats(settings: TrainingSettings) {
  const width = settings.hiddenSize;
  const parameters =
    2 * 258 * width +
    settings.sequenceLength * width +
    settings.layers * (12 * width * width + 13 * width) +
    2 * width +
    258;
  return {
    parameters,
    description: `Width ${width} · ${settings.layers} layers · ${settings.heads} heads · ${settings.sequenceLength}-token context`,
  };
}

export function modelStats(status: Pick<StudioStatus, 'model' | 'job' | 'running'>) {
  const { model, job, running } = status;
  if (!model) {
    return {
      architecture: '—',
      architectureDetail: 'No model loaded',
      parameters: '—',
      parametersDetail: 'No model loaded',
      vocabulary: '—',
      vocabularyUnit: 'tokens',
      vocabularyDetail: 'No model loaded',
      steps: '—',
      stepsDetail: 'No model loaded',
    };
  }

  const legacy = model.architecture === 'rnn';
  const architecture = legacy ? 'Character RNN' : 'Transformer';
  const architectureDetail = legacy
    ? `Width ${model.hiddenSize} · legacy checkpoint`
    : [
        `Width ${model.hiddenSize}`,
        model.layers === undefined ? undefined : `${model.layers} layers`,
        model.heads === undefined ? undefined : `${model.heads} heads`,
        model.contextSize === undefined ? undefined : `${model.contextSize}-token context`,
      ]
        .filter(Boolean)
        .join(' · ');

  return {
    architecture,
    architectureDetail,
    parameters: model.parameters.toLocaleString(),
    parametersDetail: 'Current model · trainable weights',
    vocabulary: model.vocabulary.toLocaleString(),
    vocabularyUnit: model.tokenizer === 'characters' ? 'characters' : 'tokens',
    vocabularyDetail:
      model.tokenizer === 'characters' ? 'Character tokenizer' : 'UTF-8 byte tokenizer',
    steps: model.steps.toLocaleString(),
    stepsDetail: job
      ? `${running ? 'Training' : 'Session'}: ${job.completed.toLocaleString()} / ${job.total.toLocaleString()} updates`
      : 'Total updates in loaded model',
  };
}
