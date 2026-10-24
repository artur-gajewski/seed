import { useEffect, useLayoutEffect, useRef } from 'react';
import type { StudioStatus } from '../../shared/api.ts';
import type { ChatState } from '../hooks/useChat.ts';
import { Icon } from './Icon.tsx';

interface ChatPageProps {
  chat: ChatState;
  status: StudioStatus;
  connected: boolean | null;
  studioBusy: boolean;
  onOpenStudio: () => void;
}

const SUGGESTED_PROMPTS = ['Hello!', 'Tell me a story.', 'How do you learn?'];

export function ChatPage({ chat, status, connected, studioBusy, onOpenStudio }: ChatPageProps) {
  const composer = useRef<HTMLInputElement>(null);
  const conversation = useRef<HTMLDivElement>(null);
  const canSend = Boolean(status.model && connected && !studioBusy && !chat.isSending);
  const characterCount = Array.from(chat.draft.trim()).length;

  useLayoutEffect(() => {
    const container = conversation.current;
    if (!container) {
      return;
    }

    const scrollToBottom = () => {
      container.scrollTop = container.scrollHeight;
    };

    scrollToBottom();

    // Keep the latest message visible when text wraps or the composer is resized.
    const observer = new ResizeObserver(scrollToBottom);
    observer.observe(container);
    if (container.firstElementChild) {
      observer.observe(container.firstElementChild);
    }

    return () => observer.disconnect();
  }, [chat.messages, chat.isSending]);

  useEffect(() => {
    if (!chat.isSending) {
      composer.current?.focus({ preventScroll: true });
    }
  }, [chat.isSending]);

  return (
    <section className="chat-page" aria-label="Chat with your model">
      <div className="chat-toolbar">
        <div className="chat-model">
          <Icon name="leaf" />
          <div>
            Seed <span>{status.model ? 'Loaded model' : 'No model loaded'}</span>
          </div>
          <span className={`dot ${status.model && connected ? 'active' : ''}`} />
        </div>
        <button
          className="secondary-button"
          onClick={chat.newConversation}
          disabled={chat.isSending}
        >
          <Icon name="plus" size={15} /> New chat
        </button>
      </div>

      <div
        ref={conversation}
        className={`chat-conversation ${chat.messages.length ? '' : 'is-empty'}`}
      >
        {chat.messages.length === 0 ? (
          <div className="chat-welcome">
            <span className="chat-emblem">
              <Icon name="leaf" size={34} />
            </span>
            <h1>Chat</h1>
            <p>
              {status.model
                ? 'Enter a message to generate a reply.'
                : 'Train or load a model before sending a message.'}
            </p>
            {!status.model ? (
              <button className="secondary-button" onClick={onOpenStudio}>
                Train or load a model <Icon name="arrow" size={15} />
              </button>
            ) : (
              <div className="chat-suggestions">
                {SUGGESTED_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => {
                      chat.setDraft(prompt);
                      composer.current?.focus({ preventScroll: true });
                    }}
                    disabled={chat.isSending}
                  >
                    {prompt}
                    <Icon name="arrow" size={14} />
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div
            className="chat-messages"
            role="log"
            aria-label="Conversation"
            aria-live="polite"
            aria-relevant="additions"
          >
            {chat.messages.map((message) => (
              <article key={message.id} className={`chat-message ${message.role}`}>
                <div className="chat-message-heading">
                  {message.role === 'assistant' && <Icon name="leaf" size={17} />}
                  <span>{message.role === 'user' ? 'You' : 'Seed'}</span>
                </div>
                <div className="chat-message-content">{message.content}</div>
                {message.metadata && (
                  <div className="chat-message-meta">
                    Step {message.metadata.modelStep.toLocaleString()} <span>·</span>{' '}
                    {message.metadata.elapsedMs} ms
                  </div>
                )}
              </article>
            ))}
            {chat.isSending && (
              <div className="chat-thinking" role="status">
                <Icon name="leaf" size={17} />
                <span className="dot active pulse" /> Seed is generating…
              </div>
            )}
          </div>
        )}
      </div>

      <div className="chat-composer-area">
        {chat.error && (
          <div className="alert" role="alert">
            {chat.error}
          </div>
        )}
        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
            if (canSend && characterCount <= 2000) void chat.sendMessage();
          }}
        >
          <label className="sr-only" htmlFor="chat-prompt">
            Message your trained model
          </label>
          <input
            type="text"
            id="chat-prompt"
            ref={composer}
            placeholder={
              status.model ? 'Message Seed…' : 'Train or load a model to start chatting…'
            }
            value={chat.draft}
            onChange={(event) => chat.setDraft(event.target.value)}
            disabled={!status.model || chat.isSending}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <div className="chat-composer-actions">
            <label>
              Temperature{' '}
              <select
                value={chat.temperature}
                onChange={(event) => chat.setTemperature(Number(event.target.value))}
                disabled={chat.isSending}
              >
                <option value={0.4}>0.4</option>
                <option value={0.8}>0.8</option>
                <option value={1.1}>1.1</option>
              </select>
            </label>
            <div>
              <span className={characterCount > 2000 ? 'chat-limit-exceeded' : ''}>
                {characterCount.toLocaleString()} / 2,000
              </span>
              <button
                type="submit"
                className="chat-send"
                aria-label="Send message"
                disabled={!canSend || characterCount === 0 || characterCount > 2000}
              >
                <Icon name="arrow" size={20} />
              </button>
            </div>
          </div>
        </form>
        <p className="chat-footnote">
          <span>Enter to send</span>
        </p>
      </div>
    </section>
  );
}
