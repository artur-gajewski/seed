import { useEffect, useState } from 'react';
import type { GenerateRequest, StudioStatus, TrainingSettings } from '../shared/api.ts';
import type { GenerationResult, StudioTab } from './types.ts';
import { api } from './lib/api.ts';
import { Sidebar } from './components/Sidebar.tsx';
import { StudioHeader } from './components/StudioHeader.tsx';
import { ModelStats } from './components/ModelStats.tsx';
import { CorpusEditor } from './components/CorpusEditor.tsx';
import { Playground } from './components/Playground.tsx';
import { LearningCurve } from './components/LearningCurve.tsx';
import { TrainingControls } from './components/TrainingControls.tsx';
import { CheckpointControls } from './components/CheckpointControls.tsx';
import { StudioNote } from './components/StudioNote.tsx';
import { ChatPage } from './components/ChatPage.tsx';
import { NewLifePage } from './components/NewLifePage.tsx';
import { useChat } from './hooks/useChat.ts';

export function App() {
  const chat = useChat();
  const [status, setStatus] = useState<StudioStatus>({
    model: null,
    running: false,
    history: [],
    job: null,
    error: null,
  });
  const [text, setText] = useState('');
  const [tab, setTab] = useState<StudioTab>('train');
  const [settings, setSettings] = useState<TrainingSettings>({
    hiddenSize: 64,
    steps: 1000,
    sequenceLength: 64,
    learningRate: 0.001,
    layers: 2,
    heads: 4,
    batchSize: 2,
  });
  const [generation, setGeneration] = useState<GenerateRequest>({
    prompt: 'User: Hello!\nAssistant:',
    temperature: 0.8,
    maxTokens: 240,
    topK: 0,
  });
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const next = await api('status');
        if (live) {
          setStatus(next);
          setConnected(true);
        }
      } catch {
        if (live) {
          setConnected(false);
        }
      } finally {
        if (live) {
          timer = setTimeout(poll, 700);
        }
      }
    };
    poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, []);

  async function act(fn: () => Promise<void>) {
    setError('');
    setNotice('');
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const field = (key: keyof TrainingSettings, value: string) =>
    setSettings((s) => ({ ...s, [key]: Number(value) }));

  const train = (resume: boolean) =>
    act(async () => {
      setStatus(await api('train', { text, ...settings, resume }));
    });

  const checkpoint = (action: 'save' | 'load' | 'delete') =>
    act(async () => {
      if (action === 'load') {
        const loaded = await api('checkpoint/load', {});
        setStatus(loaded);
        setSettings((current) => ({ ...current, ...loaded.trainingSettings }));
        setResult(null);
        setNotice('Checkpoint loaded. Ready to generate or resume training.');
      } else {
        const data = await api(action === 'delete' ? 'checkpoint/delete' : 'checkpoint/save', {});
        setNotice(data.message);
      }
      setStatus(await api('status'));
    });

  const importCorpus = (file: File) =>
    act(async () => {
      if (file.size > 10_000_000) {
        throw new Error('File exceeds 10 MB');
      }
      setText(await file.text());
    });

  const generate = () =>
    act(async () => {
      setResult({
        ...(await api('generate', generation)),
        prompt: generation.prompt,
      });
    });

  const stopTraining = () =>
    act(async () => {
      setStatus(await api('stop', {}));
    });

  return (
    <div className="shell">
      <Sidebar
        tab={tab}
        connected={connected}
        architecture={status.model?.architecture}
        onTabChange={setTab}
      />
      <main>
        <StudioHeader tab={tab} status={status} />
        {(error || status.error || connected === false) && (
          <div className="alert" role="alert">
            {error || status.error || 'Cannot reach the API. Start the app with npm run dev.'}
          </div>
        )}
        {notice && (
          <div className="notice" role="status">
            {notice}
          </div>
        )}
        <div hidden={tab !== 'new-life'}>
          <NewLifePage />
        </div>
        {tab === 'new-life' ? null : tab === 'chat' ? (
          <ChatPage
            chat={chat}
            status={status}
            connected={connected}
            studioBusy={busy}
            onOpenStudio={() => setTab('train')}
          />
        ) : (
          <>
            <ModelStats model={status.model} />
            <div className="content-grid">
              <div className="primary">
                {tab === 'train' ? (
                  <CorpusEditor
                    text={text}
                    running={status.running}
                    busy={busy || chat.isSending}
                    onTextChange={setText}
                    onImport={importCorpus}
                  />
                ) : (
                  <Playground
                    request={generation}
                    result={result}
                    busy={busy || chat.isSending}
                    canGenerate={Boolean(status.model && connected)}
                    onChange={setGeneration}
                    onGenerate={generate}
                  />
                )}
                <LearningCurve status={status} tab={tab} />
              </div>
              <div className="secondary">
                <TrainingControls
                  hasCorpus={text.trim().length > 0}
                  status={status}
                  settings={settings}
                  busy={busy || chat.isSending}
                  connected={connected}
                  onSettingChange={field}
                  onTrain={train}
                  onStop={stopTraining}
                />
                <CheckpointControls
                  checkpoint={status.checkpoint}
                  busy={busy || chat.isSending}
                  hasModel={Boolean(status.model)}
                  running={status.running}
                  onCheckpoint={checkpoint}
                />
                <StudioNote tab={tab} onTabChange={setTab} />
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
