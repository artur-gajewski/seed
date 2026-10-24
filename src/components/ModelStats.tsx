import type { StudioStatus, TrainingSettings } from '../../shared/api.ts';
import { modelStats, selectedModelStats } from '../lib/modelStats.ts';

export function ModelStats({
  status,
  settings,
}: {
  status: StudioStatus;
  settings?: TrainingSettings;
}) {
  const stats = modelStats(status);
  const model = status.model;
  const preview =
    settings !== undefined &&
    !status.running &&
    (!model ||
      model.architecture === 'rnn' ||
      model.hiddenSize !== settings.hiddenSize ||
      model.layers !== settings.layers ||
      model.heads !== settings.heads ||
      model.contextSize !== settings.sequenceLength);
  const selected = settings ? selectedModelStats(settings) : null;
  return (
    <section className="stats" aria-label="Current model and training progress">
      <div>
        <span>{preview ? 'SELECTED ARCHITECTURE' : 'ARCHITECTURE'}</span>
        <strong>{preview ? 'Transformer' : stats.architecture}</strong>
        <p>{preview ? selected!.description : stats.architectureDetail}</p>
      </div>
      <div>
        <span>{preview ? 'SELECTED PARAMETERS' : 'PARAMETERS'}</span>
        <strong>
          {preview ? selected!.parameters.toLocaleString() : stats.parameters}
          <small>trainable</small>
        </strong>
        <p>{preview ? 'Estimated size for selected architecture' : stats.parametersDetail}</p>
      </div>
      <div>
        <span>{preview ? 'SELECTED VOCABULARY' : 'VOCABULARY'}</span>
        <strong>
          {preview ? '258' : stats.vocabulary}
          <small>{preview ? 'tokens' : stats.vocabularyUnit}</small>
        </strong>
        <p>{preview ? 'New model · UTF-8 byte tokenizer' : stats.vocabularyDetail}</p>
      </div>
      <div>
        <span>{preview ? 'CURRENT MODEL STEPS' : 'TRAINING STEPS'}</span>
        <strong>
          {stats.steps}
          {status.model && <small>total updates</small>}
        </strong>
        <p>{stats.stepsDetail}</p>
      </div>
    </section>
  );
}
