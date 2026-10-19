import { chatGenerationRequest, extractAssistantReply, parseChatRequest } from './chat.ts';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { encodeCheckpoint, decodeCheckpoint } from './checkpoint.ts';
import type {
  ChatResponse,
  GenerateResponse,
  LossPoint,
  StudioStatus,
  TrainingJob,
  TrainingSettings,
} from '../shared/api.ts';
import { LanguageModel } from './model.ts';
import { TrainingTiming } from './training-timing.ts';
import type { ModelCheckpoint, TrainableModel } from './model.ts';
import { getErrorMessage, parseGenerationRequest, parseTrainingRequest } from './validation.ts';
import type { JsonObject } from './validation.ts';

interface TrainingSession {
  model: TrainableModel;
  job: TrainingJob;
  tokens: number[];
  validationTokens: number[];
  smoothedLoss: number | null;
  timing: TrainingTiming;
  onProgress?: (job: Readonly<TrainingJob>) => void;
}

interface SavedCheckpoint {
  model: ModelCheckpoint;
  history?: LossPoint[];
  trainingSettings?: TrainingSettings;
}

const TRAINING_BLOCK_DURATION_MS = 25;
const LOSS_HISTORY_INTERVAL = 10;
const CHECKPOINT_INTERVAL = 100;
const MAX_HISTORY_POINTS = 2000;
const LOSS_SMOOTHING_DECAY = 0.95;

// Owns the single local model. HTTP routing stays independent of training and storage.
export class ModelStudio {
  private readonly checkpointPath: string;
  private model: TrainableModel | null = null;
  private isTraining = false;
  private currentJob: TrainingJob | null = null;
  private lossHistory: LossPoint[] = [];
  private lastError: string | null = null;
  private trainingSettings: TrainingSettings | undefined;

  constructor(checkpointPath: string) {
    this.checkpointPath = checkpointPath;
  }

  getStatus(): StudioStatus {
    const model = this.model;

    return {
      trainingSettings: model
        ? {
            ...this.trainingSettings,
            hiddenSize: model.hiddenSize,
            ...(model instanceof LanguageModel
              ? {
                  layers: model.config.layers,
                  heads: model.config.heads,
                  sequenceLength: Math.min(
                    this.trainingSettings?.sequenceLength ?? model.contextSize,
                    model.contextSize,
                  ),
                }
              : {}),
          }
        : undefined,
      running: this.isTraining,
      model: model
        ? {
            hiddenSize: model.hiddenSize,
            vocabulary: model.vocabularySize,
            parameters: model.parameterCount,
            steps: model.trainingSteps,
            architecture: model instanceof LanguageModel ? 'transformer' : 'rnn',
            tokenizer: model instanceof LanguageModel ? 'utf8-bytes' : 'characters',
            contextSize: model instanceof LanguageModel ? model.contextSize : undefined,
            layers: model instanceof LanguageModel ? model.config.layers : undefined,
            heads: model instanceof LanguageModel ? model.config.heads : undefined,
          }
        : null,
      job: this.currentJob,
      history: this.lossHistory,
      error: this.lastError,
    };
  }

  startTraining(body: JsonObject, onProgress?: (job: Readonly<TrainingJob>) => void): void {
    if (this.isTraining) {
      throw new Error('Training is already running');
    }

    const settings = parseTrainingRequest(body);
    const model = settings.resume
      ? this.model
      : new LanguageModel(settings.text, settings.hiddenSize, 42, {
          layers: settings.layers,
          heads: settings.heads,
          contextSize: settings.sequenceLength,
        });

    if (!model) {
      throw new Error('No model to resume');
    }

    // Validate the corpus against the existing vocabulary before replacing any state.
    const allTokens =
      model instanceof LanguageModel
        ? model.encodeTrainingText(settings.text)
        : model.encode(settings.text);
    // A contiguous 10% holdout avoids overlapping train/validation windows.
    const splitPosition = Math.floor(allTokens.length * 0.9);
    const tokens = allTokens.slice(0, splitPosition);
    const validationTokens = allTokens.slice(splitPosition);
    const job: TrainingJob = {
      completed: 0,
      total: settings.steps,
      startedAt: Date.now(),
      loss: null,
      tokens: allTokens.length,
      trainingTokens: tokens.length,
      validationTokens: validationTokens.length,
      batchSize: settings.batchSize,
      sequenceLength: settings.sequenceLength,
      learningRate: settings.learningRate,
    };

    this.model = model;
    this.trainingSettings = {
      hiddenSize: model.hiddenSize,
      steps: settings.steps,
      sequenceLength: settings.sequenceLength,
      learningRate: settings.learningRate,
      layers: model instanceof LanguageModel ? model.config.layers : settings.layers,
      heads: model instanceof LanguageModel ? model.config.heads : settings.heads,
      batchSize: settings.batchSize,
    };
    this.currentJob = job;
    this.lastError = null;
    this.isTraining = true;

    if (!settings.resume) {
      this.lossHistory = [];
    }

    const session: TrainingSession = {
      model,
      job,
      tokens,
      validationTokens,
      smoothedLoss: null,
      timing: new TrainingTiming(),
      onProgress,
    };
    setTimeout(() => this.runTrainingBlock(session), 0);
  }

  stopTraining(): void {
    this.isTraining = false;

    if (this.currentJob) {
      this.currentJob.finishedAt = Date.now();
    }
  }

  private async runTrainingBlock(session: TrainingSession): Promise<void> {
    // A callback from a stopped job must never update a replacement model or job.
    if (!this.isTraining || this.currentJob !== session.job) {
      return;
    }

    try {
      const deadline = performance.now() + TRAINING_BLOCK_DURATION_MS;

      do {
        this.trainOneWindow(session);
        if (session.model.trainingSteps % CHECKPOINT_INTERVAL === 0) {
          await this.saveCheckpoint();
          if (!this.isTraining || this.currentJob !== session.job) {
            return;
          }
        }
      } while (
        this.isTraining &&
        session.job.completed < session.job.total &&
        performance.now() < deadline
      );

      if (session.job.completed >= session.job.total) {
        this.stopTraining();
      } else {
        // Yield between blocks so status, generation, and stop requests can run.
        setTimeout(() => this.runTrainingBlock(session), 0);
      }
    } catch (error) {
      if (this.currentJob !== session.job) {
        return;
      }
      this.lastError = getErrorMessage(error);
      this.isTraining = false;
    }
  }

  private trainOneWindow(session: TrainingSession): void {
    const { model, job, tokens } = session;
    const contextLimit = model instanceof LanguageModel ? model.contextSize : job.sequenceLength;
    const windowLength = Math.min(job.sequenceLength, contextLimit, tokens.length - 1);
    const batch = Array.from({ length: job.batchSize ?? 1 }, () => {
      const startPosition = Math.floor(Math.random() * (tokens.length - windowLength));
      return {
        inputs: tokens.slice(startPosition, startPosition + windowLength),
        targets: tokens.slice(startPosition + 1, startPosition + windowLength + 1),
      };
    });
    // Warm up, then reduce the learning rate with a cosine schedule.
    const warmupSteps = Math.max(1, Math.min(100, Math.floor(job.total * 0.05)));
    const progress = Math.max(
      0,
      (job.completed - warmupSteps) / Math.max(1, job.total - warmupSteps),
    );
    const multiplier =
      job.completed < warmupSteps
        ? (job.completed + 1) / warmupSteps
        : 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress));
    const learningRate = job.learningRate * multiplier;
    job.effectiveLearningRate = learningRate;
    const loss =
      model instanceof LanguageModel
        ? model.trainBatch(batch, learningRate)
        : model.train(batch[0].inputs, batch[0].targets, learningRate);

    session.smoothedLoss =
      session.smoothedLoss === null
        ? loss
        : LOSS_SMOOTHING_DECAY * session.smoothedLoss + 0.05 * loss;
    job.completed++;
    job.loss = session.smoothedLoss;

    const shouldRecordLoss =
      job.completed === 1 ||
      job.completed % LOSS_HISTORY_INTERVAL === 0 ||
      model.trainingSteps % CHECKPOINT_INTERVAL === 0 ||
      job.completed === job.total;

    if (shouldRecordLoss) {
      if (
        model instanceof LanguageModel &&
        (job.completed === 1 || job.completed % 100 === 0 || job.completed === job.total)
      ) {
        job.validationLoss = this.evaluateValidation(model, session.validationTokens);
      }
      this.lossHistory.push({
        step: model.trainingSteps,
        loss: session.smoothedLoss,
        validationLoss: job.validationLoss,
      });

      if (this.lossHistory.length > MAX_HISTORY_POINTS) {
        this.lossHistory.shift();
      }
    }
    job.estimatedRemainingMs = session.timing.recordUpdate(job.total - job.completed);
    session.onProgress?.(job);
  }

  private evaluateValidation(model: LanguageModel, tokens: number[]): number {
    const windowLength = Math.min(model.contextSize, tokens.length - 1);
    const windowCount = Math.min(3, Math.max(1, Math.floor((tokens.length - 1) / windowLength)));
    let totalLoss = 0;
    for (let window = 0; window < windowCount; window++) {
      const start = Math.floor(
        (window * (tokens.length - windowLength - 1)) / Math.max(1, windowCount - 1),
      );
      totalLoss += model.evaluate(
        tokens.slice(start, start + windowLength),
        tokens.slice(start + 1, start + windowLength + 1),
      );
    }
    return totalLoss / windowCount;
  }

  generate(body: JsonObject): GenerateResponse {
    if (!this.model) {
      throw new Error('Train or load a model first');
    }

    const { prompt, maxTokens, temperature, topK } = parseGenerationRequest(body);
    const startedAt = performance.now();
    const generated =
      this.model instanceof LanguageModel
        ? this.model.generateDetailed(prompt, maxTokens, temperature, topK)
        : {
            completion: this.model.generate(prompt, maxTokens, temperature, topK),
            tokens: maxTokens,
          };

    return {
      completion: generated.completion,
      tokens: generated.tokens,
      elapsedMs: Math.round(performance.now() - startedAt),
      modelStep: this.model.trainingSteps,
    };
  }

  chat(body: JsonObject): ChatResponse {
    if (!this.model) {
      throw new Error('Train or load a model first');
    }

    const request = parseChatRequest(body);
    const { completion, ...metadata } = this.generate({ ...chatGenerationRequest(request) });
    return { reply: extractAssistantReply(completion), ...metadata };
  }

  async getCheckpointInfo(): Promise<StudioStatus['checkpoint']> {
    try {
      const info = await stat(this.checkpointPath);
      return { savedAt: info.mtime.toISOString(), sizeBytes: info.size };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
      throw error;
    }
  }

  async saveCheckpoint(): Promise<void> {
    if (!this.model) {
      throw new Error('No model to save');
    }

    // Serialize before awaiting I/O so training can continue without changing this snapshot.
    const contents = encodeCheckpoint({
      model: this.model.serialize(),
      history: this.lossHistory,
      trainingSettings: this.trainingSettings,
    });
    const temporaryPath = `${this.checkpointPath}.${crypto.randomUUID()}.tmp`;

    await mkdir(path.dirname(this.checkpointPath), { recursive: true });
    try {
      await writeFile(temporaryPath, contents);
      // Rename atomically: readers see either the old checkpoint or the complete new one.
      await rename(temporaryPath, this.checkpointPath);
    } finally {
      await rm(temporaryPath, { force: true });
    }
  }

  async deleteCheckpoint(): Promise<void> {
    await rm(this.checkpointPath, { force: true });
  }

  async loadCheckpoint(): Promise<void> {
    this.requireTrainingStopped();
    const contents = await readFile(this.checkpointPath);
    const saved = decodeCheckpoint(contents) as SavedCheckpoint;

    // A training request might have arrived while the file was being read.
    this.requireTrainingStopped();
    const model = LanguageModel.restore(saved.model);
    const settings = saved.trainingSettings
      ? parseTrainingRequest({ ...saved.trainingSettings, text: ' '.repeat(64) })
      : undefined;
    this.model = model;
    if (settings) {
      const { text: _text, resume: _resume, ...trainingSettings } = settings;
      this.trainingSettings = trainingSettings;
    } else {
      this.trainingSettings = undefined;
    }
    this.lossHistory = saved.history ?? [];
    this.currentJob = null;
    this.lastError = null;
  }

  private requireTrainingStopped(): void {
    if (this.isTraining) {
      throw new Error('Stop training before loading a checkpoint');
    }
  }
}
