import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { randomBytes, randomUUID } from 'node:crypto';
import { LanguageModel } from './model.ts';
import { encodeCheckpoint, decodeCheckpoint } from './checkpoint.ts';
import type { ModelCheckpoint } from './model.ts';
import { NEW_LIFE_PRESETS } from '../shared/api.ts';
import type { NewLifeSize, NewLifeReply, NewLifeStatus } from '../shared/api.ts';
import type { JsonObject } from './validation.ts';
import { extractAssistantReply } from './chat.ts';

interface Lesson {
  message: string;
  answer?: string;
}
interface SavedLife {
  version: 1;
  model: ModelCheckpoint;
  lessons: Lesson[];
}

function briefText(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim() || Buffer.byteLength(value.trim()) > 100) {
    throw new Error(`${label} must contain 1 to 100 UTF-8 bytes`);
  }
  if (/[\r\n]|(?:User|Assistant):/.test(value)) {
    throw new Error(`${label} must be one line without conversation role labels`);
  }
  return value.trim();
}

// A separate model and replay memory; ordinary studio checkpoints are never accessed.
export class NewLife {
  private model: LanguageModel | null = null;
  private lessons: Lesson[] = [];
  private loaded = false;
  private busy = false;

  private readonly checkpointPath: string;

  private readonly size: NewLifeSize;
  private readonly legacyCheckpointPath?: string;

  constructor(checkpointPath: string, size: NewLifeSize = 'small', legacyCheckpointPath?: string) {
    this.size = size;
    this.checkpointPath = checkpointPath;
    this.legacyCheckpointPath = legacyCheckpointPath;
  }

  private async exclusive<T>(action: () => Promise<T>): Promise<T> {
    if (this.busy) throw new Error('New life is learning. Try again in a moment.');
    this.busy = true;
    try {
      if (!this.loaded) {
        let contents: Buffer | undefined;
        let migrate = false;
        try {
          contents = await readFile(this.checkpointPath);
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
          if (this.legacyCheckpointPath) {
            try {
              contents = await readFile(this.legacyCheckpointPath);
              migrate = true;
            } catch (legacyError) {
              if ((legacyError as NodeJS.ErrnoException).code !== 'ENOENT') throw legacyError;
            }
          }
        }
        if (contents !== undefined) {
          const saved = decodeCheckpoint(contents) as SavedLife;
          if (saved.version !== 1 || !Array.isArray(saved.lessons) || saved.lessons.length > 1000) {
            throw new Error('Invalid New life checkpoint');
          }
          const lessons = saved.lessons.map((lesson) => ({
            message: briefText(lesson.message, 'Message'),
            ...(lesson.answer === undefined ? {} : { answer: briefText(lesson.answer, 'Answer') }),
          }));
          const model = LanguageModel.restore(saved.model);
          if (!(model instanceof LanguageModel)) throw new Error('New life requires a Transformer');
          const preset = NEW_LIFE_PRESETS[this.size];
          if (
            Object.entries(model.config).some(
              ([key, value]) => value !== preset[key as keyof typeof model.config],
            )
          ) {
            throw new Error('New life checkpoint does not match the selected model size');
          }
          if (migrate) await rename(this.legacyCheckpointPath!, this.checkpointPath);
          this.model = model;
          this.lessons = lessons;
        }
        this.loaded = true;
      }
      return await action();
    } finally {
      this.busy = false;
    }
  }

  private status(): NewLifeStatus {
    return {
      ready: this.model !== null,
      steps: this.model?.trainingSteps ?? 0,
      lessons: this.lessons.length,
      corrections: this.lessons.filter((lesson) => lesson.answer !== undefined).length,
    };
  }

  async getStatus(): Promise<NewLifeStatus> {
    return this.exclusive(async () => this.status());
  }

  async message(body: JsonObject): Promise<NewLifeReply> {
    const message = briefText(body.message, 'Message');
    return this.exclusive(async () => {
      await this.learn({ message });
      const generated = this.model!.generate(`User: ${message}\nAssistant:`, 80, 0.8, 0);
      return { reply: extractAssistantReply(generated), status: this.status() };
    });
  }

  async teach(body: JsonObject): Promise<NewLifeStatus> {
    const message = briefText(body.message, 'Message');
    const answer = briefText(body.answer, 'Answer');
    return this.exclusive(async () => {
      await this.learn({ message, answer });
      return this.status();
    });
  }

  private async learn(lesson: Lesson): Promise<void> {
    // Work on a candidate so failed training or saving leaves the last saved life intact.
    const preset = NEW_LIFE_PRESETS[this.size];
    const model = this.model
      ? LanguageModel.restore(this.model.serialize())
      : new LanguageModel('', preset.hiddenSize, randomBytes(4).readUInt32LE(), {
          layers: preset.layers,
          heads: preset.heads,
          contextSize: preset.contextSize,
        });
    if (!(model instanceof LanguageModel)) throw new Error('New life requires a Transformer');
    const lessons = this.lessons.filter((previous) => previous.message !== lesson.message);
    const previous = this.lessons.find((entry) => entry.message === lesson.message);
    // Repeating a question must not erase an answer the user already taught.
    lessons.push(lesson.answer === undefined && previous?.answer ? previous : lesson);
    if (lessons.length > 1000) throw new Error('New life memory is full (1,000 lessons).');
    const newest = lessons.at(-1)!;
    const tokens = (entry: Lesson) =>
      model.encodeTrainingText(
        entry.answer === undefined
          ? `User: ${entry.message}`
          : `User: ${entry.message}\nAssistant: ${entry.answer}`,
      );
    for (let step = 0; step < 12; step++) {
      const batch = [
        newest,
        ...Array.from(
          { length: preset.batchSize - 1 },
          () => lessons[Math.floor(Math.random() * lessons.length)],
        ),
      ];
      model.trainBatch(
        batch.map((entry) => {
          const encoded = tokens(entry);
          return { inputs: encoded.slice(0, -1), targets: encoded.slice(1) };
        }),
        0.001,
      );
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
    const temporary = `${this.checkpointPath}.${randomUUID()}.tmp`;
    try {
      await mkdir(path.dirname(this.checkpointPath), { recursive: true });
      await writeFile(
        temporary,
        encodeCheckpoint({ version: 1, model: model.serialize(), lessons }),
      );
      await rename(temporary, this.checkpointPath);
    } finally {
      await rm(temporary, { force: true });
    }
    this.model = model;
    this.lessons = lessons;
  }
}
