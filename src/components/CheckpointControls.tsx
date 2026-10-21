import { useRef, useState } from 'react';
import { Icon } from './Icon.tsx';
import type { StudioStatus } from '../../shared/api.ts';
interface CheckpointControlsProps {
  checkpoint: StudioStatus['checkpoint'];
  busy: boolean;
  hasModel: boolean;
  running: boolean;
  onCheckpoint: (action: 'save' | 'load' | 'delete', fileName?: string) => Promise<void>;
  fileName?: string;
  files?: string[];
}

export function CheckpointControls({
  checkpoint,
  busy,
  hasModel,
  running,
  onCheckpoint,
  fileName,
  files = [],
}: CheckpointControlsProps) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [selectedFile, setSelectedFile] = useState('');

  return (
    <section className="card checkpoint">
      <div className="card-title">
        <div>
          <Icon name="save" />
          <h2>Checkpoints</h2>
        </div>
      </div>
      <p>Save or load model weights and optimizer state.</p>
      <label>
        Load a checkpoint
        <select
          value={selectedFile}
          disabled={busy || running}
          onChange={(event) => setSelectedFile(event.target.value)}
        >
          <option value="">{fileName ?? 'checkpoint.pt'} (current)</option>
          {files
            .filter((name) => name !== fileName)
            .map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
        </select>
      </label>
      <p>Save and delete apply to {fileName ?? 'checkpoint.pt'}.</p>
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
          'No saved checkpoint.'
        ) : (
          'Loading checkpoint details…'
        )}
      </p>
      <div className="checkpoint-actions">
        <button
          className="secondary-button"
          disabled={busy || !hasModel}
          onClick={() => onCheckpoint('save')}
        >
          Save checkpoint
        </button>
        <button
          className="secondary-button"
          disabled={busy || running}
          onClick={async () => {
            await onCheckpoint('load', selectedFile || undefined);
            setSelectedFile('');
          }}
        >
          Load checkpoint
        </button>
        <button
          className="secondary-button danger-button"
          disabled={busy}
          onClick={() => dialog.current?.showModal()}
        >
          Delete checkpoint
        </button>
      </div>
      <dialog
        ref={dialog}
        className="confirmation-dialog"
        aria-labelledby="delete-checkpoint-title"
        aria-describedby="delete-checkpoint-description"
      >
        <h2 id="delete-checkpoint-title">Delete saved checkpoint?</h2>
        <p id="delete-checkpoint-description">
          This permanently removes the saved checkpoint. The model currently loaded in memory will
          remain available, and you can save it again.
        </p>
        <form method="dialog" className="confirmation-actions">
          <button className="secondary-button" autoFocus>
            Cancel
          </button>
          <button
            className="secondary-button danger-button"
            disabled={busy}
            onClick={() => {
              void onCheckpoint('delete');
            }}
          >
            Delete checkpoint
          </button>
        </form>
      </dialog>
    </section>
  );
}
