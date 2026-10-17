import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { NewLifeStatus } from '../../shared/api.ts';
import { api } from '../lib/api.ts';
import { parseNewLifeInput } from '../lib/newLifeInput.ts';
import { Icon } from './Icon.tsx';

interface Exchange {
  id: number;
  message: string;
  reply: string;
}
const byteLength = (text: string) => new TextEncoder().encode(text.trim()).length;

export function NewLifePage() {
  const [status, setStatus] = useState<NewLifeStatus | null>(null);
  const [messages, setMessages] = useState<Exchange[]>([]);
  const [draft, setDraft] = useState('');
  const [answer, setAnswer] = useState('');
  const [teaching, setTeaching] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const log = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const answerInput = useRef<HTMLTextAreaElement>(null);
  const focusComposerAfterAction = useRef(false);

  useLayoutEffect(() => {
    if (teaching !== null) answerInput.current?.focus({ preventScroll: true });
  }, [teaching]);

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

    // Follow both new rows and layout changes, including teaching forms and tab visibility.
    const observer = new ResizeObserver(scrollToLatest);
    observer.observe(container);
    if (container.firstElementChild) observer.observe(container.firstElementChild);
    return () => observer.disconnect();
  }, [messages, busy, teaching, error, notice]);

  async function act(action: () => Promise<void>) {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      await action();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  const valid = (text: string) => byteLength(text) > 0 && byteLength(text) <= 100;
  const isTeachingDraft = draft.trim().startsWith('!');
  const draftContent = isTeachingDraft ? draft.trim().slice(1).trim() : draft.trim();

  async function teachAnswer(message: string, preferredAnswer: string) {
    setStatus(await api('new-life/teach', { message, answer: preferredAnswer.trim() }));
    focusComposerAfterAction.current = true;
    setTeaching(null);
    setAnswer('');
    setNotice('Answer learned and saved. Ask again to see how Seed responds.');
  }

  return (
    <section className="chat-page new-life-page" aria-label="New life">
      <div className="chat-toolbar">
        <div className="chat-model">
          <Icon name="leaf" />
          <div>
            New life <span>{status?.ready ? 'Growing with you' : 'A fresh beginning'}</span>
          </div>
        </div>
        <button
          className="secondary-button"
          disabled={busy}
          onClick={() => {
            setMessages([]);
            setTeaching(null);
            setAnswer('');
            setNotice('');
          }}
        >
          New conversation
        </button>
      </div>
      <div className="new-life-intro">
        <h1>A little more with every lesson.</h1>
        <p>
          This Seed starts from scratch. It learns language from your messages and answers from your
          teaching. Early replies will be gibberish. Use “Teach an answer” to show it what you
          wanted to hear.
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
        <p className="new-life-note">
          Learning is saved automatically in its own checkpoint. Previous lessons are revisited with
          each message. Generated replies are only learned when you teach them as an answer. New
          conversations keep its learning; the visible conversation clears when you reload.
        </p>
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
                <button
                  className="secondary-button"
                  disabled={busy}
                  onClick={() => {
                    setTeaching(exchange.id);
                    setAnswer('');
                    setNotice('');
                  }}
                >
                  Teach an answer
                </button>
                {teaching === exchange.id && (
                  <form
                    className="new-life-teacher"
                    onSubmit={(event) => {
                      event.preventDefault();
                      if (busy || !valid(answer)) return;
                      void act(async () => {
                        await teachAnswer(exchange.message, answer);
                      });
                    }}
                  >
                    <label htmlFor={`lesson-${exchange.id}`}>What should Seed say?</label>
                    <textarea
                      id={`lesson-${exchange.id}`}
                      ref={answerInput}
                      rows={2}
                      value={answer}
                      disabled={busy}
                      onChange={(event) => setAnswer(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
                          event.preventDefault();
                          event.currentTarget.form?.requestSubmit();
                        }
                      }}
                    />
                    <small>{byteLength(answer)} / 100 bytes · one short line</small>
                    <div className="new-life-teacher-actions">
                      <button
                        className="secondary-button"
                        type="button"
                        disabled={busy}
                        onClick={() => setTeaching(null)}
                      >
                        Cancel
                      </button>
                      <button className="primary-button" disabled={busy || !valid(answer)}>
                        Teach and save
                      </button>
                    </div>
                  </form>
                )}
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
        {notice && (
          <div className="notice" role="status">
            {notice}
          </div>
        )}
        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
            if (busy || !valid(draftContent)) return;
            focusComposerAfterAction.current = true;
            void act(async () => {
              const input = parseNewLifeInput(draft, messages.at(-1)?.message);
              if (input.kind === 'teach') {
                await teachAnswer(input.message, input.answer);
                setDraft('');
                return;
              }
              const { message } = input;
              const result = await api('new-life/message', { message });
              setStatus(result.status);
              setMessages((previous) => [
                ...previous,
                { id: Date.now(), message, reply: result.reply },
              ]);
              setDraft('');
            });
          }}
        >
          <label className="sr-only" htmlFor="new-life-message">
            Message New life
          </label>
          <textarea
            id="new-life-message"
            ref={composer}
            rows={2}
            placeholder="Say something simple…"
            value={draft}
            disabled={busy}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
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
          Start with ! to teach an answer to your latest message, for example: !The sky is blue.
        </p>
      </div>
    </section>
  );
}
