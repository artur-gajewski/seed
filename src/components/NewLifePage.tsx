import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { NewLifeStatus } from '../../shared/api.ts';
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

export function NewLifePage() {
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
    let live = true;
    api('new-life/status').then(
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
  }, []);

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

  const valid = (text: string) => byteLength(text) > 0 && byteLength(text) <= 100;
  const { isTeaching: isTeachingDraft, content: draftContent } = newLifeInputContent(draft);

  async function teachAnswer(message: string, preferredAnswer: string, correction: string) {
    setStatus(await api('new-life/teach', { message, answer: preferredAnswer.trim() }));
    focusComposerAfterAction.current = true;
    const reply = TEACHING_REPLIES[Math.floor(Math.random() * TEACHING_REPLIES.length)];
    setMessages((previous) => [
      ...previous,
      { id: Date.now(), message: correction, reply, teaching: true },
    ]);
  }

  return (
    <section className="chat-page new-life-page" aria-label="New life">
      <div className="chat-toolbar">
        <div className="chat-model">
          <Icon name="leaf" />
          <div>New life {!status?.ready && <span>A fresh beginning</span>}</div>
        </div>
        <button
          className="secondary-button"
          disabled={busy}
          onClick={() => {
            setMessages([]);
          }}
        >
          New conversation
        </button>
      </div>
      <div className="new-life-intro">
        <p>
          This model starts from scratch. It learns language from your messages and answers from
          your teaching. Early replies will be gibberish.
          <br />
          Type ! followed by the answer you wanted,
          or a correction such as “No, the sky is blue.”
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
            <p className="chat-thinking" role="status">
              Seed is learning and saving…
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
              );
              if (input.kind === 'teach') {
                await teachAnswer(input.message, input.answer, draft.trim());
                rememberInput(draft);
                return;
              }
              const { message } = input;
              const result = await api('new-life/message', { message });
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
              {byteLength(draftContent)} / 100 bytes ·{' '}
              {isTeachingDraft ? 'Teaching latest answer' : 'one short line'}
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
