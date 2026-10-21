import type { StudioStatus } from '../../shared/api.ts';
import type { StudioTab } from '../types.ts';
interface StudioHeaderProps {
  tab: StudioTab;
  status: StudioStatus;
}

export function StudioHeader({ tab, status }: StudioHeaderProps) {
  return (
    <>
      <header>
        <div className="breadcrumb">
          Workspace <span>/</span>{' '}
          <b>
            {tab === 'train'
              ? 'Training'
              : tab === 'chat'
                ? 'Chat'
                : tab === 'new-life'
                  ? 'New Life'
                  : 'Text generation'}
          </b>
        </div>
      </header>
      {tab !== 'chat' && tab !== 'new-life' && (
        <div className="page-heading">
          <div>
            <h1>{tab === 'train' ? 'Train a model' : 'Generate text'}</h1>
            <p>
              {tab === 'train'
                ? 'Import or paste text, choose settings, and start training.'
                : 'Enter a prompt to generate a continuation.'}
            </p>
          </div>
          <div
            className={`state-badge ${status.running ? 'training' : status.model ? 'ready' : ''}`}
          >
            <span>
              {status.running
                ? 'Training in progress'
                : status.model
                  ? 'Model ready'
                  : 'No model loaded'}
            </span>
          </div>
        </div>
      )}
    </>
  );
}
