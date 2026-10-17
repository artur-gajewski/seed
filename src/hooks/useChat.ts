import { useRef, useState } from 'react';
import type { ChatMessage, ChatResponse } from '../../shared/api.ts';
import { api } from '../lib/api.ts';

export interface ConversationMessage extends ChatMessage {
  id: string;
  metadata?: Omit<ChatResponse, 'reply'>;
}

export function useChat() {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState('');
  const [temperature, setTemperature] = useState(0.8);
  const requestInFlight = useRef(false);

  async function sendMessage(): Promise<void> {
    if (requestInFlight.current || !draft.trim()) {
      return;
    }

    const userMessage: ConversationMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: draft.trim(),
    };
    const nextMessages = [...messages, userMessage];
    requestInFlight.current = true;
    setIsSending(true);
    setError('');
    setMessages(nextMessages);
    setDraft('');

    try {
      // Send the newest 20 complete turns plus the current user message.
      const context = nextMessages.slice(-41).map(({ role, content }) => ({ role, content }));
      const { reply, ...metadata } = await api('chat', {
        messages: context,
        maxTokens: 400,
        temperature,
        topK: 0,
      });

      if (!reply.trim()) {
        throw new Error('The model returned an empty reply. Try again or train it for more steps.');
      }

      setMessages([
        ...nextMessages,
        { id: crypto.randomUUID(), role: 'assistant', content: reply, metadata },
      ]);
    } catch (error) {
      // Restore the unsent prompt so retrying cannot create consecutive user turns.
      setMessages(messages);
      setDraft(userMessage.content);
      setError(error instanceof Error ? error.message : String(error));
    } finally {
      requestInFlight.current = false;
      setIsSending(false);
    }
  }

  function newConversation(): void {
    if (requestInFlight.current) {
      return;
    }

    setMessages([]);
    setDraft('');
    setError('');
  }

  return {
    messages,
    draft,
    setDraft,
    isSending,
    error,
    temperature,
    setTemperature,
    sendMessage,
    newConversation,
  };
}

export type ChatState = ReturnType<typeof useChat>;
