import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import {
  access,
  copyFile,
  link,
  mkdtemp,
  readFile,
  readdir,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeCheckpoint } from './checkpoint.ts';
import { chatGenerationRequest, extractAssistantReply, parseChatRequest } from './chat.ts';
import { parseGenerationRequest, parseTrainingRequest } from './validation.ts';
import type { JsonObject } from './validation.ts';
import { NEW_LIFE_PRESETS } from '../shared/api.ts';
import type {
  StudioStatus,
  GenerateResponse,
  ChatResponse,
  NewLifeSize,
  NewLifeStatus,
  NewLifeReply,
} from '../shared/api.ts';

const root = fileURLToPath(new URL('../', import.meta.url));
interface WorkerMessage {
  id?: number;
  event?: string;
  version?: number;
  result?: unknown;
  error?: string;
  status?: StudioStatus;
}

export class TorchStudio {
  private child: ChildProcessWithoutNullStreams | null = null;
  private ready: Promise<void> | null = null;
  private serial = 0;
  private pending = new Map<
    number,
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();
  private status: StudioStatus = {
    running: false,
    model: null,
    job: null,
    history: [],
    error: null,
  };
  private closing = false;
  private closePromise: Promise<void> | null = null;
  private mutating = false;
  private failure: Error | null = null;
  private options: { python?: string; device?: string };
  private checkpointPath: string;
  private lifeBusy = new Set<NewLifeSize>();
  private loadedLives = new Set<NewLifeSize>();

  constructor(
    checkpointPath = path.join(root, 'data/models/seed_default.pt'),
    options: { python?: string; device?: string } = {},
  ) {
    this.checkpointPath = path.resolve(checkpointPath);
    this.options = options;
  }

  private ensureReady(): Promise<void> {
    if (this.closing) return Promise.reject(new Error('PyTorch worker is shutting down'));
    if (this.failure) return Promise.reject(this.failure);
    if (this.ready) return this.ready;
    const python =
      this.options.python ||
      process.env.SEED_PYTHON ||
      (existsSync(path.join(root, '.venv/bin/python'))
        ? path.join(root, '.venv/bin/python')
        : 'python3');
    this.ready = new Promise<void>((resolve, reject) => {
      let stderr = '';
      let settled = false;
      const child = spawn(python, ['-u', '-m', 'seed_torch.worker'], {
        cwd: root,
        env: {
          ...process.env,
          SEED_DEVICE: this.options.device ?? process.env.SEED_DEVICE ?? 'auto',
          PYTHONPATH: [path.join(root, 'python'), process.env.PYTHONPATH]
            .filter(Boolean)
            .join(path.delimiter),
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      this.child = child;
      const startupTimer = setTimeout(() => {
        fail(
          new Error(
            'PyTorch startup timed out. Restart the server and check the Python installation.',
          ),
        );
        child.kill('SIGKILL');
      }, 60000);
      startupTimer.unref();
      const fail = (error: Error) => {
        clearTimeout(startupTimer);
        if (this.child !== child) return;
        this.child = null;
        this.ready = null;
        this.failure = error;
        this.loadedLives.clear();
        this.status = { ...this.status, running: false, model: null, error: error.message };
        for (const request of this.pending.values()) request.reject(error);
        this.pending.clear();
        if (!settled) {
          settled = true;
          reject(error);
        }
      };
      child.on('error', (error) =>
        fail(
          new Error(
            `Cannot start PyTorch: ${error.message}. Install python dependencies; see docs/pytorch.md.`,
          ),
        ),
      );
      child.stdin.on('error', (error) => fail(error));
      child.stderr.on('data', (chunk: Buffer) => {
        stderr = (stderr + chunk.toString()).slice(-4000);
      });
      child.on('exit', (code, signal) =>
        fail(new Error(`PyTorch worker exited (${signal ?? code}). ${stderr.trim()}`)),
      );
      const lines = createInterface({ input: child.stdout });
      lines.on('line', (line) => {
        try {
          const message = JSON.parse(line) as WorkerMessage;
          if (message.status) this.status = message.status;
          if (message.event === 'ready') {
            if (message.version !== 1) throw new Error('Unsupported PyTorch worker protocol');
            settled = true;
            clearTimeout(startupTimer);
            resolve();
          }
          if (message.id !== undefined) {
            const request = this.pending.get(message.id);
            this.pending.delete(message.id);
            if (message.error) request?.reject(new Error(message.error));
            else request?.resolve(message.result);
          }
        } catch (error) {
          fail(error instanceof Error ? error : new Error(String(error)));
          child.kill();
        }
      });
    });
    return this.ready;
  }

  private async request<T>(action: string, body: unknown = {}): Promise<T> {
    await this.ensureReady();
    if (this.pending.size >= 32) throw new Error('PyTorch is busy. Try again in a moment.');
    return new Promise<T>((resolve, reject) => {
      const id = ++this.serial;
      this.pending.set(id, { resolve: (value) => resolve(value as T), reject });
      this.child!.stdin.write(JSON.stringify({ id, action, body }) + '\n', (error) => {
        if (error) {
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  async getStatus(): Promise<StudioStatus> {
    if (!this.failure) await this.ensureReady();
    const directory = path.dirname(this.checkpointPath);
    const files = await readdir(directory).catch((error: NodeJS.ErrnoException) => {
      if (error.code === 'ENOENT') return [];
      throw error;
    });
    return {
      ...this.status,
      checkpoint: await this.getCheckpointInfo(),
      checkpointFileName: path.basename(this.checkpointPath),
      checkpoints: files
        .filter((name) => name.endsWith('.pt') && !name.startsWith('new-life-'))
        .sort(),
    };
  }

  private async mutate<T>(action: () => Promise<T>): Promise<T> {
    if (this.mutating)
      throw new Error('A model operation is already in progress. Try again in a moment.');
    this.mutating = true;
    try {
      return await action();
    } finally {
      this.mutating = false;
    }
  }

  async startTraining(body: JsonObject): Promise<StudioStatus> {
    return this.mutate(async () => {
      const settings = parseTrainingRequest(body);
      const device = body.device;
      if (device !== undefined && !['auto', 'cpu', 'mps', 'cuda'].includes(String(device)))
        throw new Error('Invalid training device');
      await this.request('train', {
        settings: { ...settings, ...(device ? { device } : {}) },
        path: this.checkpointPath,
      });
      return this.getStatus();
    });
  }

  async stopTraining(): Promise<StudioStatus> {
    return this.mutate(async () => {
      await this.request('stop');
      return this.getStatus();
    });
  }

  async generate(body: JsonObject): Promise<GenerateResponse> {
    return this.request('generate', parseGenerationRequest(body));
  }

  async chat(body: JsonObject): Promise<ChatResponse> {
    const { completion, ...metadata } = await this.generate({
      ...chatGenerationRequest(parseChatRequest(body)),
    });
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
    return this.mutate(async () => {
      await this.request('save');
    });
  }
  async createCheckpoint(body: JsonObject): Promise<StudioStatus> {
    return this.mutate(async () => {
      if (typeof body.name !== 'string' || !body.name.trim())
        throw new Error('Enter a checkpoint name');
      const name = body.name.trim();
      const fileName = name.endsWith('.pt') ? name : `${name}.pt`;
      if (
        !/^[A-Za-z0-9][\w.-]*\.pt$/.test(fileName) ||
        fileName.length > 103 ||
        fileName.startsWith('new-life-')
      ) {
        throw new Error(
          'Use up to 100 letters, numbers, dots, hyphens or underscores for the checkpoint name',
        );
      }
      if (
        typeof body.settings !== 'object' ||
        body.settings === null ||
        Array.isArray(body.settings)
      )
        throw new Error('Supply training settings');
      const {
        text: _text,
        resume: _resume,
        ...settings
      } = parseTrainingRequest({ ...(body.settings as JsonObject), text: ' '.repeat(64) });
      if (
        body.device !== undefined &&
        !['auto', 'cpu', 'mps', 'cuda'].includes(String(body.device))
      )
        throw new Error('Invalid training device');
      const target = path.join(path.dirname(this.checkpointPath), fileName);
      await this.request('create', {
        path: target,
        settings,
        ...(body.device ? { device: body.device } : {}),
      });
      this.checkpointPath = target;
      return this.getStatus();
    });
  }
  async deleteCheckpoint(): Promise<void> {
    return this.mutate(async () => {
      await this.request('delete', { path: this.checkpointPath });
    });
  }

  private async withInterchange<T>(
    legacyPath: string,
    action: (file: string, saved: JsonObject) => Promise<T>,
  ): Promise<T> {
    const saved = decodeCheckpoint(await readFile(legacyPath)) as JsonObject;
    const directory = await mkdtemp(path.join(tmpdir(), 'seed-torch-import-'));
    try {
      const file = path.join(directory, 'checkpoint.json');
      await writeFile(file, JSON.stringify(saved));
      return await action(file, saved);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  }

  async loadCheckpoint(body: JsonObject = {}): Promise<StudioStatus> {
    return this.mutate(async () => {
      let target = this.checkpointPath;
      if (body.fileName !== undefined) {
        const name = body.fileName;
        if (typeof name !== 'string' || !/^[\w.-]+\.pt$/.test(name) || name.startsWith('new-life-'))
          throw new Error('Choose a studio .pt checkpoint filename');
        target = path.join(path.dirname(this.checkpointPath), name);
      }
      try {
        await access(target);
        await this.request('load', { path: target });
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT' || body.fileName !== undefined)
          throw error;
        const previousDefault = path.join(path.dirname(target), 'checkpoint.pt');
        if (path.basename(target) === 'seed_default.pt' && existsSync(previousDefault)) {
          const temporary = `${target}.${crypto.randomUUID()}.tmp`;
          try {
            await copyFile(previousDefault, temporary);
            try {
              await link(temporary, target);
            } catch (copyError) {
              if ((copyError as NodeJS.ErrnoException).code !== 'EEXIST') throw copyError;
            }
          } finally {
            await rm(temporary, { force: true });
          }
          await this.request('load', { path: target });
          this.checkpointPath = target;
          return this.getStatus();
        }
        const legacy =
          path.basename(target) === 'seed_default.pt' &&
          !existsSync(target.replace(/\.pt$/, '.json'))
            ? path.join(path.dirname(target), 'checkpoint.json')
            : target.replace(/\.pt$/, '.json');
        await this.withInterchange(legacy, (file) =>
          this.request('load', { path: target, interchange: file }),
        );
      }
      this.checkpointPath = target;
      return this.getStatus();
    });
  }

  async life(
    action: 'status' | 'message' | 'teach',
    body: JsonObject = {},
  ): Promise<NewLifeStatus | NewLifeReply> {
    const size = body.size ?? 'small';
    if (typeof size !== 'string' || !Object.hasOwn(NEW_LIFE_PRESETS, size))
      throw new Error('Invalid New life model size');
    const selected = size as NewLifeSize;
    if (this.lifeBusy.has(selected))
      throw new Error('New life is learning. Try again in a moment.');
    const brief = (value: unknown, label: string) => {
      if (typeof value !== 'string' || !value.trim() || Buffer.byteLength(value.trim()) > 100)
        throw new Error(`${label} must contain 1 to 100 UTF-8 bytes`);
      if (/[\r\n]|(?:User|Assistant):/.test(value))
        throw new Error(`${label} must be one line without conversation role labels`);
      return value.trim();
    };
    const lesson =
      action === 'status'
        ? undefined
        : {
            message: brief(body.message, 'Message'),
            ...(action === 'teach' ? { answer: brief(body.answer, 'Answer') } : {}),
          };
    this.lifeBusy.add(selected);
    try {
      const preset = NEW_LIFE_PRESETS[selected];
      const target = path.join(
        path.dirname(this.checkpointPath),
        `new-life-checkpoint-${preset.label}.pt`,
      );
      const payload = {
        action,
        size,
        path: target,
        lesson,
        config: {
          hiddenSize: preset.hiddenSize,
          layers: preset.layers,
          heads: preset.heads,
          contextSize: preset.contextSize,
        },
        batchSize: preset.batchSize,
      };
      const run = async (extra = {}) => {
        const result = await this.request<NewLifeStatus | (NewLifeReply & { completion: string })>(
          'life',
          { ...payload, ...extra },
        );
        this.loadedLives.add(selected);
        if ('completion' in result)
          return { status: result.status, reply: extractAssistantReply(result.completion) };
        return result;
      };
      if (!this.loadedLives.has(selected) && !existsSync(target)) {
        const suffix = selected === 'small' ? '' : `-${selected}`;
        const legacy = [
          target.replace(/\.pt$/, '.json'),
          path.join(path.dirname(target), `new-life-checkpoint${suffix}.json`),
        ].find(existsSync);
        if (legacy)
          return await this.withInterchange(legacy, async (file, saved) => {
            if (saved.version !== 1 || !Array.isArray(saved.lessons) || saved.lessons.length > 1000)
              throw new Error('Invalid New life checkpoint');
            const lessons = saved.lessons.map((entry: { message: unknown; answer?: unknown }) => ({
              message: brief(entry.message, 'Message'),
              ...(entry.answer === undefined ? {} : { answer: brief(entry.answer, 'Answer') }),
            }));
            return run({ interchange: file, lessons });
          });
      }
      return await run();
    } finally {
      this.lifeBusy.delete(selected);
    }
  }

  async close(): Promise<void> {
    if (!this.closePromise) this.closePromise = this.shutdown();
    return this.closePromise;
  }

  private async shutdown(): Promise<void> {
    if (!this.child) return;
    const child = this.child;
    const response = this.request('shutdown');
    this.closing = true;
    const timer = setTimeout(() => child.kill('SIGKILL'), 30000);
    timer.unref();
    try {
      await response;
    } finally {
      child.stdin.end();
      if (child.exitCode === null && child.signalCode === null)
        await new Promise<void>((resolve) => child.once('exit', () => resolve()));
      clearTimeout(timer);
    }
  }
}
