import type { StudioStatus } from '../../shared/api.ts';
import type { StudioTab } from '../types.ts';
import { LossChart } from './LossChart.tsx';

export function LearningCurve({ status, tab }: { status: StudioStatus; tab: StudioTab }) {
  return (
    <section className="card learning">
      <div className="card-title">
        <div>
          <span className="step-label">{tab === 'train' ? '03' : '02'}</span>
          <h2>Training loss</h2>
        </div>
        <span className="loss-key">
          <i /> Training loss
        </span>
      </div>
      <div className="loss-meta">
        <strong>
          {status.job?.loss?.toFixed(3) ?? status.history.at(-1)?.loss.toFixed(3) ?? '—'}
        </strong>
        <span>
          cross-entropy <i>·</i> lower is better
        </span>
      </div>
      <div className="validation-metrics">
        <span>
          Validation loss:{' '}
          <strong>
            {(status.job?.validationLoss ?? status.history.at(-1)?.validationLoss)?.toFixed(3) ??
              '—'}
          </strong>
        </span>
        <span>Dashed line · held-out loss</span>
        {status.trainingSettings?.trainingMode === 'supervised-qa' && (
          <>
            <span>
              Training answer accuracy:{' '}
              {(
                (status.job?.trainingAccuracy ?? status.history.at(-1)?.trainingAccuracy ?? 0) * 100
              ).toFixed(1)}
              %
            </span>
            <span>
              Validation answer accuracy:{' '}
              {(
                (status.job?.validationAccuracy ?? status.history.at(-1)?.validationAccuracy ?? 0) *
                100
              ).toFixed(1)}
              %
            </span>
            <span>
              Training evaluation loss:{' '}
              {(status.job?.trainingLoss ?? status.history.at(-1)?.trainingLoss)?.toFixed(3) ?? '—'}
            </span>
          </>
        )}
      </div>
      <LossChart points={status.history} />
      <div className="chart-caption">
        Training and validation cross-entropy in nats. Rising validation loss while training loss
        falls can indicate overfitting. Repeated examples across the split can make validation look
        too optimistic.
      </div>
    </section>
  );
}
