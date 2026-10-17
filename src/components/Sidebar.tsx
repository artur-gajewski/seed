import { Icon } from './Icon.tsx';
import type { StudioTab } from '../types.ts';
interface SidebarProps {
  tab: StudioTab;
  connected: boolean | null;
  architecture?: 'transformer' | 'rnn';
  onTabChange: (tab: StudioTab) => void;
}

export function Sidebar({ tab, connected, architecture, onTabChange }: SidebarProps) {
  return (
    <aside className="sidebar">
      <a href="/" className="brand">
        <span className="brand-mark">
          <Icon name="leaf" size={27} />
        </span>
        seed<span className="brand-dot">.</span>
      </a>
      <div className="workspace-label">Navigation</div>
      <button
        className={`nav ${tab === 'train' ? 'active' : ''}`}
        onClick={() => onTabChange('train')}
        aria-label="Training"
      >
        <Icon name="layers" /> Training
      </button>
      <button
        className={`nav ${tab === 'generate' ? 'active' : ''}`}
        onClick={() => onTabChange('generate')}
        aria-label="Text generation"
      >
        <Icon name="spark" /> Text generation <Icon name="arrow" size={14} />
      </button>
      <button
        className={`nav ${tab === 'chat' ? 'active' : ''}`}
        onClick={() => onTabChange('chat')}
        aria-label="Chat"
        aria-current={tab === 'chat' ? 'page' : undefined}
      >
        <Icon name="chat" /> Chat
      </button>
      <div className="engine-status">
        <span className={`dot ${connected ? 'active' : ''}`} />
        <div>
          {connected ? 'Local engine connected' : 'Connecting to engine…'}
          <small>{architecture === 'rnn' ? 'Character RNN' : 'Transformer'} · CPU</small>
        </div>
      </div>
    </aside>
  );
}
