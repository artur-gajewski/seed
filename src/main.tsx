import { createRoot } from 'react-dom/client';
import { App } from './App.tsx';
import './style.css';

const root = document.getElementById('root');
if (!root) {
  throw new Error('Missing root element');
}
createRoot(root).render(<App />);
