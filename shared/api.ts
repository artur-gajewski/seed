export interface LossPoint {
  step: number;
  loss: number;
  validationLoss?: number;
}
export interface TrainingSettings {
  hiddenSize: number;
  steps: number;
  sequenceLength: number;
  learningRate: number;
  layers: number;
  heads: number;
  batchSize: number;
}
export interface TrainRequest extends TrainingSettings {
  text: string;
  resume: boolean;
  device?: 'auto' | 'cpu' | 'mps' | 'cuda';
}
export interface TrainingJob {
  completed: number;
  total: number;
  startedAt: number;
  finishedAt?: number;
  estimatedRemainingMs?: number;
  loss: number | null;
  validationLoss?: number;
  trainingTokens?: number;
  validationTokens?: number;
  batchSize?: number;
  effectiveLearningRate?: number;
  tokens: number;
  sequenceLength: number;
  learningRate: number;
}
export interface StudioStatus {
  backend?: {
    engine: 'pytorch';
    device: string;
    precision: 'float32';
    devices: ('cpu' | 'mps' | 'cuda')[];
  };
  checkpointFileName?: string;
  checkpoints?: string[];
  trainingSettings?: Partial<TrainingSettings>;
  checkpoint?: { savedAt: string; sizeBytes: number } | null;
  running: boolean;
  model: {
    checkpointFileName?: string;
    hiddenSize: number;
    vocabulary: number;
    parameters: number;
    steps: number;
    architecture?: 'transformer' | 'rnn';
    contextSize?: number;
    layers?: number;
    heads?: number;
    tokenizer?: 'utf8-bytes' | 'characters';
  } | null;
  job: TrainingJob | null;
  history: LossPoint[];
  error: string | null;
}
export interface GenerateRequest {
  prompt: string;
  maxTokens: number;
  temperature: number;
  topK: number;
}
export interface GenerateResponse {
  completion: string;
  tokens: number;
  elapsedMs: number;
  modelStep: number;
}
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
export interface ChatRequest {
  messages: ChatMessage[];
  maxTokens: number;
  temperature: number;
  topK: number;
}
export interface ChatResponse {
  reply: string;
  tokens: number;
  elapsedMs: number;
  modelStep: number;
}
export const NEW_LIFE_PRESETS = {
  small: { label: '38K', hiddenSize: 32, layers: 1, heads: 4, batchSize: 2, contextSize: 256 },
  medium: { label: '496K', hiddenSize: 128, layers: 2, heads: 4, batchSize: 4, contextSize: 256 },
  large: { label: '2.6M', hiddenSize: 256, layers: 3, heads: 8, batchSize: 6, contextSize: 256 },
  'x-large': {
    label: '13M',
    hiddenSize: 512,
    layers: 4,
    heads: 8,
    batchSize: 8,
    contextSize: 256,
  },
} as const;
export type NewLifeSize = keyof typeof NEW_LIFE_PRESETS;
export interface NewLifeStatus {
  steps: number;
  lessons: number;
  corrections: number;
  ready: boolean;
}
export interface NewLifeReply {
  reply: string;
  status: NewLifeStatus;
}
export interface ApiResponses {
  'new-life/status': NewLifeStatus;
  'new-life/settings': NewLifeStatus;
  'new-life/message': NewLifeReply;
  'new-life/teach': NewLifeStatus;
  status: StudioStatus;
  train: StudioStatus;
  stop: StudioStatus;
  generate: GenerateResponse;
  chat: ChatResponse;
  'checkpoint/save': { message: string };
  'checkpoint/load': StudioStatus;
  'checkpoint/create': StudioStatus;
  'checkpoint/delete': { message: string };
}
export interface ApiRequests {
  'new-life/status': undefined;
  'new-life/settings': { size: NewLifeSize };
  'new-life/message': { message: string; size?: NewLifeSize };
  'new-life/teach': { message: string; answer: string; size?: NewLifeSize };
  status: undefined;
  train: TrainRequest;
  stop: Record<string, never>;
  generate: GenerateRequest;
  chat: ChatRequest;
  'checkpoint/save': Record<string, never>;
  'checkpoint/load': { fileName?: string };
  'checkpoint/create': {
    name: string;
    settings: TrainingSettings;
    device?: TrainRequest['device'];
  };
  'checkpoint/delete': Record<string, never>;
}
export type ApiRoute = keyof ApiResponses;
