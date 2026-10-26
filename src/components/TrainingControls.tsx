import { useRef } from 'react';
import type { StudioStatus, TrainingSettings, TrainRequest } from '../../shared/api.ts';
import { Icon } from './Icon.tsx';
import { formatDuration } from '../lib/formatDuration.ts';
interface TrainingControlsProps {
  device: NonNullable<TrainRequest['device']>;
  onDeviceChange: (device: NonNullable<TrainRequest['device']>) => void;
  hasCorpus: boolean;
  status: StudioStatus;
  settings: TrainingSettings;
  busy: boolean;
  connected: boolean | null;
  onSettingChange: (key: keyof TrainingSettings, value: string) => void;
  onTrain: (resume: boolean) => Promise<void>;
  onStop: () => Promise<void>;
}

export function TrainingControls({
  hasCorpus,
  status,
  settings,
  busy,
  connected,
  onSettingChange,
  onTrain,
  onStop,
  device,
  onDeviceChange,
}: TrainingControlsProps) {
  const helpDialog = useRef<HTMLDialogElement>(null);
  const progress = status.job ? (status.job.completed / status.job.total) * 100 : 0;
  return (
    <section className="card config">
      <div className="card-title">
        <div>
          <span className="step-label">02</span>
          <h2>Training settings</h2>
          <button
            type="button"
            className="help-button"
            aria-label="About training settings"
            aria-haspopup="dialog"
            aria-controls="training-settings-help"
            onClick={() => helpDialog.current?.showModal()}
          >
            ?
          </button>
        </div>
      </div>
      <dialog
        ref={helpDialog}
        id="training-settings-help"
        className="confirmation-dialog training-help-dialog"
        aria-labelledby="training-settings-help-title"
      >
        <h2 id="training-settings-help-title">About training settings</h2>
        <div className="training-help-content">
          <p>
            These settings control the model’s size and how it learns to predict the next token from
            your training text. Tokens here are UTF-8 bytes, so a character can use more than one
            token.
          </p>
          <dl>
            <dt>Training mode</dt>
            <dd>
              Language modeling learns to predict every next token in random text windows.
              Supervised Q&amp;A uses complete User: / Assistant: examples, gives the model the full
              question as context, and learns from the assistant’s answer and end token. Keep
              assistant-only loss enabled for answer-focused training. Q&amp;A splits complete
              examples into training and validation sets and reports answer accuracy using greedy
              generation.
            </dd>
            <dt>Embedding width and layers</dt>
            <dd>Set the model’s capacity. Larger values require more memory and training time.</dd>
            <dt>Attention heads</dt>
            <dd>
              Split attention into groups that can learn different relationships in the text.
              Embedding width must be divisible by the number of heads.
            </dd>
            <dt>Batch size</dt>
            <dd>
              How many text windows (language modeling) or complete examples (Supervised Q&amp;A)
              are used for each weight update.
            </dd>
            <dt>Training steps</dt>
            <dd>How many weight updates to run. More steps give the model more practice.</dd>
            <dt>Learning rate</dt>
            <dd>
              The peak size of weight adjustments. Lower values learn more slowly; values that are
              too high can make training unstable.
            </dd>
            <dt>Context window</dt>
            <dd>
              How many tokens the model can use at once. Longer windows provide more context and
              cost more computation.
            </dd>
          </dl>
          <p>
            Suggested Q&amp;A fine-tuning: continue the existing model with learning rate 0.0001,
            batch size 16, context 512 if all examples fit, and 2,000 steps. Evaluate every 100
            updates on 100 examples. Use Top-K = 1 for greedy chat.
          </p>
          <p>
            Train with new weights applies all settings and starts with random weights. Continue
            existing model keeps its architecture and context limit, while using the selected batch
            size, steps, and learning rate.
          </p>
        </div>
        <form method="dialog" className="confirmation-actions">
          <button className="secondary-button" autoFocus>
            Close
          </button>
        </form>
      </dialog>
      <p className="card-description">Set the model size and training parameters.</p>
      <fieldset disabled={status.running || busy}>
        <label>
          Training mode
          <select
            value={settings.trainingMode ?? 'language-modeling'}
            onChange={(e) => onSettingChange('trainingMode', e.target.value)}
          >
            <option value="language-modeling">Language modeling</option>
            <option value="supervised-qa">Supervised Q&amp;A</option>
          </select>
        </label>
        {settings.trainingMode === 'supervised-qa' && (
          <div className="qa-settings">
            {(
              [
                ['assistantOnlyLoss', 'Assistant-only loss'],
                ['resetOptimizerOnModeChange', 'Reset optimizer when switching training modes'],
                ['restartSchedule', 'Start a new learning-rate schedule'],
              ] as const
            ).map(([key, label]) => (
              <label key={key} className="training-checkbox">
                <input
                  type="checkbox"
                  checked={settings[key] ?? true}
                  onChange={(e) => onSettingChange(key, String(e.target.checked))}
                />
                {label}
              </label>
            ))}
            <div className="field-pair">
              {(
                [
                  ['validationSplit', 'Validation split', 0.1, 0.01, 0.5],
                  ['randomSeed', 'Random seed', 42, 0, 4294967295],
                  ['evaluationInterval', 'Evaluation interval', 100, 1, 1000000],
                  ['evaluationSamples', 'Evaluation samples', 100, 1, 10000],
                ] as const
              ).map(([key, label, fallback, min, max]) => (
                <label key={key}>
                  {label}
                  <input
                    type="number"
                    min={min}
                    max={max}
                    step={key === 'validationSplit' ? 0.01 : 1}
                    value={settings[key] ?? fallback}
                    onChange={(e) => onSettingChange(key, e.target.value)}
                  />
                </label>
              ))}
            </div>
          </div>
        )}
        <label>
          Training device
          <select
            value={device}
            onChange={(event) =>
              onDeviceChange(event.target.value as NonNullable<TrainRequest['device']>)
            }
          >
            <option value="auto">Automatic (prefer GPU)</option>
            {(status.backend?.devices ?? ['cpu']).map((value) => (
              <option key={value} value={value}>
                {value === 'mps'
                  ? 'Apple GPU (MPS)'
                  : value === 'cuda'
                    ? 'NVIDIA GPU (CUDA)'
                    : 'CPU'}
              </option>
            ))}
          </select>
          <small className="device-status">
            {status.backend
              ? `PyTorch · ${status.backend.device.toUpperCase()} · Float32`
              : 'Connecting to PyTorch…'}
          </small>
        </label>
        <label>
          Embedding width <span>Model size</span>
          <select
            value={settings.hiddenSize}
            onChange={(e) => onSettingChange('hiddenSize', e.target.value)}
          >
            {[...new Set([16, 32, 48, 64, 128, 256, 384, 512, 768, 1024, settings.hiddenSize])]
              .sort((a, b) => a - b)
              .map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
          </select>
        </label>
        <div className="field-pair">
          <label>
            Layers
            <select
              value={settings.layers}
              onChange={(event) => onSettingChange('layers', event.target.value)}
            >
              {Array.from({ length: 12 }, (_, index) => index + 1).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            Attention heads
            <select
              value={settings.heads}
              onChange={(event) => onSettingChange('heads', event.target.value)}
            >
              {[...new Set([1, 2, 4, 8, 16, settings.heads])]
                .sort((a, b) => a - b)
                .map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
            </select>
          </label>
        </div>
        <label>
          Batch size{' '}
          <span>
            {settings.trainingMode === 'supervised-qa'
              ? 'Examples per update'
              : 'Windows per update'}
          </span>
          <select
            value={settings.batchSize}
            onChange={(event) => onSettingChange('batchSize', event.target.value)}
          >
            {[...new Set([1, 2, 4, 8, 16, 32, 64, settings.batchSize])]
              .sort((a, b) => a - b)
              .map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
          </select>
        </label>
        <div className="field-pair">
          <label>
            Training steps
            <input
              type="number"
              min="1"
              max="1000000"
              value={settings.steps}
              onChange={(e) => onSettingChange('steps', e.target.value)}
            />
          </label>
          <label>
            Learning rate
            <input
              type="number"
              min="0.00001"
              max="0.05"
              step="0.0001"
              value={settings.learningRate}
              onChange={(e) => onSettingChange('learningRate', e.target.value)}
            />
          </label>
        </div>
        <label>
          Context window <span>UTF-8 bytes</span>
          <select
            value={settings.sequenceLength}
            onChange={(e) => onSettingChange('sequenceLength', e.target.value)}
          >
            {[
              ...new Set([
                16,
                32,
                48,
                64,
                128,
                256,
                384,
                512,
                768,
                1024,
                1280,
                1536,
                1792,
                2048,
                settings.sequenceLength,
              ]),
            ]
              .sort((a, b) => a - b)
              .map((n) => (
                <option key={n} value={n}>
                  {n} tokens
                </option>
              ))}
          </select>
        </label>
      </fieldset>
      <div className="training-progress">
        <div>
          <span>{status.running ? 'Training' : status.job ? 'Progress' : 'Not started'}</span>
          <b>{Math.round(progress)}%</b>
        </div>
        <div className="track">
          <div style={{ width: `${progress}%` }} />
        </div>
        <small>
          {status.job
            ? `${status.job.completed.toLocaleString()} / ${status.job.total.toLocaleString()} updates`
            : 'Weights are created when training starts'}
        </small>
        {status.job && (
          <small className="training-time-remaining">
            Estimated time remaining:{' '}
            {status.job.completed === status.job.total
              ? '00'
              : !status.running
                ? 'Stopped'
                : status.job.estimatedRemainingMs === undefined
                  ? 'Calculating…'
                  : formatDuration(status.job.estimatedRemainingMs)}
          </small>
        )}
      </div>
      {status.running ? (
        <button className="primary-button stop" disabled={busy} onClick={onStop}>
          <Icon name="stop" /> Stop training
        </button>
      ) : (
        <button
          className="primary-button"
          disabled={busy || !connected || !hasCorpus}
          onClick={() => onTrain(false)}
        >
          <Icon name="play" />
          {status.model ? 'Train with new weights' : 'Start training'}
          <Icon name="arrow" />
        </button>
      )}
      {status.model && !status.running && (
        <button
          className="secondary-button"
          disabled={busy || !connected || !hasCorpus}
          onClick={() => onTrain(true)}
        >
          Continue existing model
        </button>
      )}
      <p className="button-note">
        {status.model
          ? 'New model resets weights. Continue uses the loaded architecture and optimizer.'
          : 'New models use a Transformer with random weights. The final 10% of the corpus is reserved for validation.'}
      </p>
    </section>
  );
}
