import type { ChatMessage, ChatRequest, GenerateRequest } from '../shared/api.ts';
import { parseGenerationRequest } from './validation.ts';
import type { JsonObject } from './validation.ts';

const MAX_CONTEXT_CHARACTERS = 4096;
const MAX_MESSAGE_CHARACTERS = 2000;
const MAX_MESSAGES = 41;

export function parseChatRequest(body: JsonObject): ChatRequest {
  if (
    !Array.isArray(body.messages) ||
    body.messages.length === 0 ||
    body.messages.length > MAX_MESSAGES
  ) {
    throw new Error(`Supply between 1 and ${MAX_MESSAGES} chat messages`);
  }

  const messages: ChatMessage[] = body.messages.map((message: unknown, index: number) => {
    if (typeof message !== 'object' || message === null) {
      throw new Error('Each message must have a role and content');
    }

    const { role, content } = message as Record<string, unknown>;
    const expectedRole = index % 2 === 0 ? 'user' : 'assistant';

    if (role !== expectedRole) {
      throw new Error('Messages must alternate user and assistant, starting with a user');
    }

    if (
      typeof content !== 'string' ||
      !content.trim() ||
      Array.from(content).length > MAX_MESSAGE_CHARACTERS
    ) {
      throw new Error(`Messages must contain 1–${MAX_MESSAGE_CHARACTERS} characters`);
    }

    return { role: expectedRole, content };
  });

  if (messages.at(-1)!.role !== 'user') {
    throw new Error('The last message must be from the user');
  }

  const settings = parseGenerationRequest({ ...body, prompt: '' });
  return {
    messages,
    maxTokens: settings.maxTokens,
    temperature: settings.temperature,
    topK: settings.topK,
  };
}

export function createChatPrompt(messages: ChatMessage[]): string {
  const format = (message: ChatMessage) =>
    `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.content}`;
  let prompt = `${format(messages.at(-1)!)}\nAssistant:`;

  // Keep recent complete turns. Never cut through the user's current message.
  for (let index = messages.length - 3; index >= 0; index -= 2) {
    const previousTurn = `${format(messages[index])}\n${format(messages[index + 1])}\n\n`;
    if (Array.from(previousTurn + prompt).length > MAX_CONTEXT_CHARACTERS) {
      break;
    }
    prompt = previousTurn + prompt;
  }

  return prompt;
}

export function chatGenerationRequest(request: ChatRequest): GenerateRequest {
  return {
    prompt: createChatPrompt(request.messages),
    maxTokens: request.maxTokens,
    temperature: request.temperature,
    topK: request.topK,
  };
}

export function extractAssistantReply(completion: string): string {
  // A small model may continue writing the next turn; display only its assistant reply.
  return completion.split(/\n(?:User|Assistant):/)[0].trim();
}
