import type { GenerateResponse } from '../shared/api.ts';

export type StudioTab = 'train' | 'generate' | 'chat' | 'new-life';
export type GenerationResult = GenerateResponse & { prompt: string };
