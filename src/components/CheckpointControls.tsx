import { useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { Icon } from './Icon.tsx';
import type { StudioStatus } from '../../shared/api.ts';
interface CheckpointControlsProps {
  checkpoint: StudioStatus['checkpoint'];
  busy: boolean;
  running: boolean;
  onLoad: (fileName?: string) => Promise<void>;
  onCreate: (name: string) => Promise<void>;
  fileName?: string;
  files?: string[];
}

export function CheckpointControls({
  checkpoint,
  busy,
  running,
  onLoad,
  onCreate,
  fileName,
  files = [],
}: CheckpointControlsProps) {
  const [selectedFile, setSelectedFile] = useState('');
  const dialog = useRef<HTMLDialogElement>(null);
  const [name, setName] = useState('');
  const [createError, setCreateError] = useState('');

  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCreateError('');
    try {
      await onCreate(name);
      setSelectedFile('');
      dialog.current?.close();
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <section className="card checkpoint">
      <div className="card-title">
        <div>
          <Icon name="save" />
          <h2>Models</h2>
        </div>
      </div>
      <p>Load model weights and optimizer state.</p>
      <label>
        Select model
        <select
          value={selectedFile}
          disabled={busy || running}
          onChange={(event) => setSelectedFile(event.target.value)}
        >
          <option value="">{fileName ?? 'seed_default.pt'} (current)</option>
          {files
            .filter((name) => name !== fileName)
            .map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
        </select>
      </label>
      <p className="checkpoint-metadata">
        {checkpoint ? (
          <>
            Last saved:{' '}
            <time dateTime={checkpoint.savedAt}>
              {new Date(checkpoint.savedAt).toLocaleString()}
            </time>
            <br />
            File size: {(checkpoint.sizeBytes / 1_000_000).toFixed(2)} MB (
            {(checkpoint.sizeBytes / 1_000_000_000).toFixed(4)} GB)
          </>
        ) : checkpoint === null ? (
          'No saved model.'
        ) : (
          'Loading model details…'
        )}
      </p>
      <div className="checkpoint-actions">
        <button
          className="secondary-button"
          disabled={busy || running}
          onClick={async () => {
            await onLoad(selectedFile || undefined);
            setSelectedFile('');
          }}
        >
          Load model
        </button>
      </div>
      <button
        className="secondary-button create-checkpoint-button"
        disabled={busy || running}
        onClick={() => {
          setName('');
          setCreateError('');
          dialog.current?.showModal();
        }}
      >
        Create new model
      </button>
      <dialog
        ref={dialog}
        className="confirmation-dialog"
        aria-labelledby="create-checkpoint-title"
        onCancel={(event) => {
          if (busy) event.preventDefault();
        }}
      >
        <h2 id="create-checkpoint-title">Create new model</h2>
        <p>Create an untrained model using the selected training settings.</p>
        <form onSubmit={create}>
          <label>
            Model name
            <input
              autoFocus
              required
              maxLength={103}
              value={name}
              onChange={(event) => setName(event.target.value)}
              disabled={busy}
              placeholder="my-experiment"
            />
          </label>
          <p>The .pt extension is added automatically.</p>
          {createError && <p role="alert">{createError}</p>}
          <div className="confirmation-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={busy}
              onClick={() => dialog.current?.close()}
            >
              Cancel
            </button>
            <button type="submit" className="primary-button" disabled={busy || !name.trim()}>
              {busy ? 'Creating…' : 'Create'}
            </button>
          </div>
        </form>
      </dialog>
    </section>
  );
}
