import { useRef } from 'react';
import { Icon } from './Icon.tsx';
import { SAMPLE } from '../data/sample.ts';
interface CorpusEditorProps {
  text: string;
  running: boolean;
  busy: boolean;
  onTextChange: (text: string) => void;
  onImport: (file: File) => Promise<void>;
}

export function CorpusEditor({ text, running, busy, onTextChange, onImport }: CorpusEditorProps) {
  const upload = useRef<HTMLInputElement>(null);
  const tokens = Array.from(text).length;
  return (
    <section className="card corpus">
      <div className="card-title">
        <div>
          <span className="step-label">01</span>
          <h2>Training data</h2>
        </div>
        <span className="muted-tag">PLAIN TEXT</span>
      </div>
      <p className="card-description">Paste text or import a UTF-8 .txt file.</p>
      <div className="editor-toolbar">
        <span>
          <span className="file-dot" /> training-corpus.txt
        </span>
        <div>
          <button
            className="text-button"
            disabled={running || busy}
            onClick={() => onTextChange(SAMPLE)}
          >
            Use sample
          </button>
          <button
            className="text-button"
            disabled={running || busy}
            onClick={() => upload.current?.click()}
          >
            <Icon name="upload" size={14} /> Import .txt
          </button>
        </div>
      </div>
      <input
        ref={upload}
        type="file"
        accept=".txt,text/plain"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            onImport(file);
          }
          e.target.value = '';
        }}
      />
      <textarea
        className="corpus-input"
        aria-label="Training corpus"
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
        disabled={running}
        spellCheck="false"
      />
      <div className="editor-footer">
        <span>
          {tokens.toLocaleString()} characters <i>·</i>{' '}
          {new TextEncoder().encode(text).length.toLocaleString()} UTF-8 bytes
        </span>
        <span>UTF-8</span>
      </div>
      <div className="tip">
        <Icon name="leaf" size={17} />
        <span>Use consistent formatting across training examples.</span>
      </div>
    </section>
  );
}
