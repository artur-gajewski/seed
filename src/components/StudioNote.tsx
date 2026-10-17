import { Icon } from './Icon.tsx';
import type { StudioTab } from '../types.ts';

export function StudioNote({
  tab,
  onTabChange,
}: {
  tab: StudioTab;
  onTabChange: (tab: StudioTab) => void;
}) {
  return (
    <div className="honest-note">
      <span>Model limits</span>
      <p>
        Reply quality depends on corpus size, examples, model size, and training time. Compare
        validation loss with training loss to check for overfitting. A small local model will not
        match a large pretrained assistant.
      </p>
      <button onClick={() => onTabChange(tab === 'train' ? 'generate' : 'train')}>
        {tab === 'train' ? 'Generate text' : 'Back to training'}
        <Icon name="arrow" size={15} />
      </button>
    </div>
  );
}
