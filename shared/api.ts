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
  'new-life/message': NewLifeReply;
  'new-life/teach': NewLifeStatus;
  status: StudioStatus;
  train: StudioStatus;
  stop: StudioStatus;
  generate: GenerateResponse;
  chat: ChatResponse;
  'checkpoint/save': { message: string };
  'checkpoint/load': StudioStatus;
  'checkpoint/delete': { message: string };
}
export interface ApiRequests {
  'new-life/status': undefined;
  'new-life/message': { message: string };
  'new-life/teach': { message: string; answer: string };
  status: undefined;
  train: TrainRequest;
  stop: Record<string, never>;
  generate: GenerateRequest;
  chat: ChatRequest;
  'checkpoint/save': Record<string, never>;
  'checkpoint/load': Record<string, never>;
  'checkpoint/delete': Record<string, never>;
}
export type ApiRoute = keyof ApiResponses;
