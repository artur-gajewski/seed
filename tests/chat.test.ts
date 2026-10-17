import test from 'node:test';
import assert from 'node:assert/strict';
import { createChatPrompt, extractAssistantReply, parseChatRequest } from '../server/chat.ts';
import { ModelStudio } from '../server/studio.ts';
import type { ChatMessage } from '../shared/api.ts';

test('chat keeps recent complete turns and the current Unicode prompt within the context limit', () => {
  const messages: ChatMessage[] = [
    { role: 'user', content: 'a'.repeat(1900) },
    { role: 'assistant', content: 'b'.repeat(1900) },
    { role: 'user', content: 'Hello!' },
    { role: 'assistant', content: 'Hello there.' },
    { role: 'user', content: '🌱'.repeat(1000) },
  ];
  const prompt = createChatPrompt(messages);
  assert.ok(Array.from(prompt).length <= 4096);
  assert.ok(prompt.startsWith('User: Hello!\nAssistant: Hello there.\n\n'));
  assert.ok(prompt.endsWith(`User: ${'🌱'.repeat(1000)}\nAssistant:`));
  assert.ok(!prompt.includes('a'.repeat(1900)));
});

test('chat validates message roles, lengths, order, and generation settings', () => {
  const message = { role: 'user', content: 'Hello!' };
  assert.deepEqual(parseChatRequest({ messages: [message] }), {
    messages: [message],
    maxTokens: 200,
    temperature: 0.8,
    topK: 0,
  });
  assert.throws(() => parseChatRequest({ messages: [] }), /between 1 and 41/);
  assert.throws(() => parseChatRequest({ messages: [null] }), /role and content/);
  assert.throws(
    () => parseChatRequest({ messages: [{ role: 'system', content: 'Hello' }] }),
    /alternate/,
  );
  assert.throws(() => parseChatRequest({ messages: [message, message] }), /alternate/);
  assert.throws(
    () => parseChatRequest({ messages: [message, { role: 'assistant', content: 'Hello' }] }),
    /last message/,
  );
  assert.throws(() => parseChatRequest({ messages: [{ ...message, content: ' ' }] }), /characters/);
  assert.throws(
    () => parseChatRequest({ messages: [{ ...message, content: '🌱'.repeat(2001) }] }),
    /characters/,
  );
  assert.throws(() => parseChatRequest({ messages: [message], temperature: 0 }), /Temperature/);
});

test('chat removes generated subsequent turns without changing the assistant text', () => {
  assert.equal(
    extractAssistantReply(' Hello!\n\nUser: Next question\nAssistant: Next reply'),
    'Hello!',
  );
  assert.equal(
    extractAssistantReply('A paragraph.\n\nAnother paragraph.'),
    'A paragraph.\n\nAnother paragraph.',
  );
});

test('chat replies come from the same trained weights used by text generation', async () => {
  const studio = new ModelStudio('/tmp/seed-chat-unused-checkpoint.json');
  assert.throws(
    () => studio.chat({ messages: [{ role: 'user', content: 'Hello!' }] }),
    /Train or load/,
  );

  studio.startTraining({
    text: 'User: Hello!\nAssistant: Hello there.\n\n'.repeat(10),
    steps: 1,
    hiddenSize: 16,
  });
  const deadline = Date.now() + 5000;
  while (studio.getStatus().running && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  studio.stopTraining();
  assert.equal(studio.getStatus().model!.steps, 1);

  const messages: ChatMessage[] = [{ role: 'user', content: 'Hello!' }];
  const settings = { maxTokens: 40, temperature: 0.8, topK: 1 };
  const expected = studio.generate({
    prompt: createChatPrompt(messages),
    ...settings,
  });
  const actual = studio.chat({ messages, ...settings });

  assert.equal(actual.reply, extractAssistantReply(expected.completion));
  assert.equal(actual.modelStep, expected.modelStep);
  assert.equal(actual.tokens, expected.tokens);
  assert.equal(
    typeof studio.chat({ messages: [{ role: 'user', content: '🦄' }], maxTokens: 1 }).reply,
    'string',
  );
});
