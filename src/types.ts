import type { GenerateResponse } from '../shared/api.ts';

export type StudioTab = 'train' | 'generate' | 'chat';
export type GenerationResult = GenerateResponse & { prompt: string };
