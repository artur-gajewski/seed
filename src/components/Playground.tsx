import type { GenerateRequest } from '../../shared/api.ts';
import type { GenerationResult } from '../types.ts';
import { Icon } from './Icon.tsx';
interface PlaygroundProps {
  request: GenerateRequest;
  result: GenerationResult | null;
  busy: boolean;
  canGenerate: boolean;
  onChange: (request: GenerateRequest) => void;
  onGenerate: () => Promise<void>;
}

export function Playground({
  request,
  result,
  busy,
  canGenerate,
  onChange,
  onGenerate,
}: PlaygroundProps) {
  const { prompt, temperature, maxTokens, topK } = request;
  return (
    <section className="card playground">
      <div className="card-title">
        <div>
          <span className="step-label">01</span>
          <h2>Prompt</h2>
        </div>
        <Icon name="spark" />
      </div>
      <p className="card-description">Generate text from the loaded model.</p>
      <label className="input-label" htmlFor="prompt">
        PROMPT
      </label>
      <textarea
        id="prompt"
        className="prompt-input"
        value={prompt}
        onChange={(e) => onChange({ ...request, prompt: e.target.value })}
        spellCheck="false"
      />
      <div className="generation-settings">
        <label>
          Temperature{' '}
          <input
            type="number"
            min="0.1"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => onChange({ ...request, temperature: Number(e.target.value) })}
          />
        </label>
        <label>
          New tokens{' '}
          <input
            type="number"
            min="1"
            max="1000"
            value={maxTokens}
            onChange={(e) => onChange({ ...request, maxTokens: Number(e.target.value) })}
          />
        </label>
        <label>
          Top K (0 = all)
          <input
            type="number"
            min="0"
            max="512"
            value={topK}
            onChange={(e) => onChange({ ...request, topK: Number(e.target.value) })}
          />
        </label>
      </div>
      <button className="primary-button" disabled={busy || !canGenerate} onClick={onGenerate}>
        <Icon name="spark" />
        {busy ? 'Generating…' : 'Generate continuation'}
        <Icon name="arrow" />
      </button>
      <div className="output">
        <div className="output-title">
          MODEL OUTPUT{' '}
          <span>
            {result
              ? `${result.tokens} tokens · ${result.elapsedMs} ms · step ${result.modelStep}`
              : 'Waiting for a prompt'}
          </span>
        </div>
        {result ? (
          <pre>
            <span>{result.prompt}</span>
            {result.completion}
          </pre>
        ) : (
          <p>No output yet.</p>
        )}
      </div>
    </section>
  );
}
