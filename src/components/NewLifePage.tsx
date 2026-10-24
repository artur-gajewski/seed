import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { NEW_LIFE_PRESETS, newLifeByteLimit } from '../../shared/api.ts';
import type { NewLifeSize, NewLifeStatus } from '../../shared/api.ts';
import { api } from '../lib/api.ts';
import { newLifeInputContent, parseNewLifeInput } from '../lib/newLifeInput.ts';
import { Icon } from './Icon.tsx';

interface Exchange {
  id: number;
  message: string;
  reply: string;
  teaching?: boolean;
}
const TEACHING_REPLIES = [
  'Ok, I will try to learn this.',
  'Thanks, I will practice that.',
  'Got it. I will keep learning from this.',
  'Thank you for teaching me. I will try to remember.',
  'Ok, I will work on getting that right.',
];
const byteLength = (text: string) => new TextEncoder().encode(text.trim()).length;
const MODEL_SIZE_STORAGE_KEY = 'seed.newLife.modelSize';

function rememberedModelSize(): NewLifeSize {
  try {
    const saved = localStorage.getItem(MODEL_SIZE_STORAGE_KEY);
    if (saved && Object.hasOwn(NEW_LIFE_PRESETS, saved)) return saved as NewLifeSize;
  } catch {
    // Browser storage can be unavailable; the default model still works.
  }
  return 'x-large';
}

export function NewLifePage() {
  const [initialSize] = useState(rememberedModelSize);
  const [size, setSize] = useState<NewLifeSize>(initialSize);
  const [status, setStatus] = useState<NewLifeStatus | null>(null);
  const [messages, setMessages] = useState<Exchange[]>([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const log = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLInputElement>(null);
  const focusComposerAfterAction = useRef(false);
  const inputHistory = useRef<string[]>([]);
  const historyPosition = useRef<number | null>(null);
  const unfinishedDraft = useRef('');

  function rememberInput(input: string) {
    inputHistory.current.push(input);
    historyPosition.current = null;
    unfinishedDraft.current = '';
    setDraft('');
  }

  useEffect(() => {
    if (!busy && focusComposerAfterAction.current) {
      focusComposerAfterAction.current = false;
      composer.current?.focus({ preventScroll: true });
    }
  }, [busy]);

  useEffect(() => {
    try {
      localStorage.setItem(MODEL_SIZE_STORAGE_KEY, size);
    } catch {
      // Keep model selection usable when browser storage is unavailable.
    }
  }, [size]);

  useEffect(() => {
    let live = true;
    api('new-life/settings', { size: initialSize }).then(
      (next) => {
        if (live) setStatus(next);
      },
      (reason: unknown) => {
        if (live) setError(String(reason));
      },
    );
    return () => {
      live = false;
    };
  }, [initialSize]);

  useLayoutEffect(() => {
    const container = log.current;
    if (!container) return;

    const scrollToLatest = () => {
      container.scrollTop = container.scrollHeight;
    };
    scrollToLatest();

    // Follow both new rows and layout changes, including tab visibility.
    const observer = new ResizeObserver(scrollToLatest);
    observer.observe(container);
    if (container.firstElementChild) observer.observe(container.firstElementChild);
    return () => observer.disconnect();
  }, [messages, busy, error]);

  async function act(action: () => Promise<void>) {
    setBusy(true);
    setError('');
    try {
      await action();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  const { isTeaching: isTeachingDraft, content: draftContent } = newLifeInputContent(draft);
  const latestMessage = messages.findLast((exchange) => !exchange.teaching)?.message;
  const byteLimit = newLifeByteLimit(size, isTeachingDraft ? latestMessage : undefined);
  const valid = (text: string) => byteLength(text) > 0 && byteLength(text) <= byteLimit;

  async function teachAnswer(message: string, preferredAnswer: string, correction: string) {
    setStatus(await api('new-life/teach', { message, answer: preferredAnswer.trim(), size }));
    focusComposerAfterAction.current = true;
    const reply = TEACHING_REPLIES[Math.floor(Math.random() * TEACHING_REPLIES.length)];
    setMessages((previous) => [
      ...previous,
      { id: Date.now(), message: correction, reply, teaching: true },
    ]);
  }

  return (
    <section className="chat-page new-life-page" aria-label="New life">
      <div className="new-life-intro">
        <div className="new-life-settings">
          <label>
            Model preset:{' '}
            <select
              value={size}
              disabled={busy}
              onChange={(event) => {
                const nextSize = event.target.value as NewLifeSize;
                void act(async () => {
                  const next = await api('new-life/settings', { size: nextSize });
                  setSize(nextSize);
                  setStatus(next);
                  setMessages([]);
                  setDraft('');
                  inputHistory.current = [];
                  historyPosition.current = null;
                  unfinishedDraft.current = '';
                });
              }}
            >
              {Object.entries(NEW_LIFE_PRESETS).map(([value, preset]) => (
                <option key={value} value={value}>
                  {preset.label}
                </option>
              ))}
            </select>
          </label>
          <span className="new-life-model-details">
            Width {NEW_LIFE_PRESETS[size].hiddenSize} · {NEW_LIFE_PRESETS[size].layers}{' '}
            {NEW_LIFE_PRESETS[size].layers === 1 ? 'layer' : 'layers'} ·{' '}
            {NEW_LIFE_PRESETS[size].heads} heads · Batch {NEW_LIFE_PRESETS[size].batchSize} ·{' '}
            {NEW_LIFE_PRESETS[size].contextSize}-token context. Each size remembers its own lessons.
          </span>
        </div>
        <p>
          This model starts from scratch. It learns language from your messages and answers from
          your teaching. Early replies will be gibberish.
        </p>
        <div className="new-life-stats" aria-live="polite">
          <span>
            <b>{status?.lessons ?? 0} / 1,000</b> remembered lessons
          </span>
          <span>
            <b>{status?.corrections ?? 0}</b> taught answers
          </span>
          <span>
            <b>{status?.steps ?? 0}</b> learning updates
          </span>
        </div>
      </div>
      <div className="chat-conversation" ref={log}>
        <div
          className="chat-messages"
          role="log"
          aria-label="New life conversation"
          aria-live="polite"
        >
          {messages.map((exchange) => (
            <div key={exchange.id}>
              <article className="chat-message user">
                <div className="chat-message-heading">You</div>
                <div className="chat-message-content">{exchange.message}</div>
              </article>
              <article className="chat-message assistant">
                <div className="chat-message-heading">
                  <Icon name="leaf" size={17} /> Seed
                </div>
                <div className="chat-message-content">
                  {exchange.reply || '(No readable reply yet.)'}
                </div>
              </article>
            </div>
          ))}
          {busy && (
            <p className="chat-thinking new-life-learning" role="status">
              <Icon name="leaf" size={22} />
              <span className="sr-only">Seed is learning and saving…</span>
            </p>
          )}
        </div>
      </div>
      <div className="chat-composer-area">
        {error && (
          <div className="alert" role="alert">
            {error}
          </div>
        )}
        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
            if (busy || !valid(draftContent)) return;
            focusComposerAfterAction.current = true;
            void act(async () => {
              const input = parseNewLifeInput(
                draft,
                messages.findLast((exchange) => !exchange.teaching)?.message,
                size,
              );
              if (input.kind === 'teach') {
                await teachAnswer(input.message, input.answer, draft.trim());
                rememberInput(draft);
                return;
              }
              const { message } = input;
              const result = await api('new-life/message', { message, size });
              setStatus(result.status);
              setMessages((previous) => [
                ...previous,
                { id: Date.now(), message, reply: result.reply },
              ]);
              rememberInput(draft);
            });
          }}
        >
          <label className="sr-only" htmlFor="new-life-message">
            Message New life
          </label>
          <input
            type="text"
            id="new-life-message"
            ref={composer}
            placeholder="Say something simple…"
            value={draft}
            disabled={busy}
            onChange={(event) => {
              historyPosition.current = null;
              setDraft(event.target.value);
            }}
            onKeyDown={(event) => {
              if (event.nativeEvent.isComposing) return;
              if (
                (event.key === 'ArrowUp' || event.key === 'ArrowDown') &&
                !event.altKey &&
                !event.ctrlKey &&
                !event.metaKey &&
                !event.shiftKey
              ) {
                event.preventDefault();
                const history = inputHistory.current;
                if (!history.length) return;
                if (event.key === 'ArrowUp') {
                  if (historyPosition.current === null) {
                    unfinishedDraft.current = draft;
                    historyPosition.current = history.length - 1;
                  } else {
                    historyPosition.current = Math.max(0, historyPosition.current - 1);
                  }
                  setDraft(history[historyPosition.current]);
                } else if (historyPosition.current !== null) {
                  if (historyPosition.current < history.length - 1) {
                    historyPosition.current++;
                    setDraft(history[historyPosition.current]);
                  } else {
                    historyPosition.current = null;
                    setDraft(unfinishedDraft.current);
                  }
                }
              } else if (event.key === 'Enter') {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <div className="chat-composer-actions">
            <span>
              {byteLength(draftContent)} / {byteLimit} bytes
              {isTeachingDraft ? ' · Teaching latest answer' : ''}
            </span>
            <button
              className="chat-send"
              aria-label={isTeachingDraft ? 'Teach and save latest answer' : 'Send and learn'}
              disabled={busy || !valid(draftContent)}
            >
              <Icon name="arrow" size={20} />
            </button>
          </div>
        </form>
        <p className="chat-footnote">
          Teach the latest answer with <code>!The sky is blue.</code> or{' '}
          <code>No, the sky is blue.</code> The send button switches to teaching when a correction
          is recognized.
        </p>
      </div>
    </section>
  );
}
