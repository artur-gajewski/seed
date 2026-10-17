import type { StudioStatus } from '../../shared/api.ts';

export function ModelStats({ model }: { model: StudioStatus['model'] }) {
  return (
    <section className="stats">
      <div>
        <span>ARCHITECTURE</span>
        <strong>
          {model?.architecture === 'rnn' ? 'Character RNN' : 'Transformer'}{' '}
          <small>{model?.architecture === 'rnn' ? 'legacy' : 'decoder'}</small>
        </strong>
        <p>
          {model?.architecture === 'rnn'
            ? 'Legacy checkpoint'
            : `${model?.layers ?? 2} layers · ${model?.heads ?? 4} attention heads`}
        </p>
      </div>
      <div>
        <span>PARAMETERS</span>
        <strong>
          {model ? model.parameters.toLocaleString() : '—'}
          <small>trainable</small>
        </strong>
        <p>Initialized from random weights</p>
      </div>
      <div>
        <span>VOCABULARY</span>
        <strong>
          {model?.vocabulary ?? '—'}
          <small>{model?.tokenizer === 'characters' ? 'characters' : 'tokens'}</small>
        </strong>
        <p>
          {model?.tokenizer === 'characters'
            ? 'One character = one token'
            : '256 UTF-8 bytes + start/stop tokens'}
        </p>
      </div>
      <div>
        <span>TRAINING STEPS</span>
        <strong>
          {(model?.steps ?? 0).toLocaleString()}
          <small>updates</small>
        </strong>
        <p>
          {model?.architecture === 'rnn' ? 'Backpropagation + Adam' : 'Backpropagation + AdamW'}
        </p>
      </div>
    </section>
  );
}
