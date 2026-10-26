import { useEffect, useRef, useState } from 'react';
import { SAMPLE } from './lib/sample.ts';
import type {
  GenerateRequest,
  StudioStatus,
  TrainingSettings,
  TrainRequest,
} from '../shared/api.ts';
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
import { ChatPage } from './components/ChatPage.tsx';
import { NewLifePage } from './components/NewLifePage.tsx';
import { useChat } from './hooks/useChat.ts';
import { useStoredState } from './hooks/useStoredState.ts';
import {
  DEFAULT_GENERATION_SETTINGS,
  isTrainingSettings,
  isGenerationSettings,
  isTrainingDevice,
} from './lib/studioPreferences.ts';
import { DEFAULT_TRAINING_SETTINGS, trainingSettingsForModel } from './lib/trainingSettings.ts';

export function App() {
  const statusRevision = useRef(0);
  const chat = useChat();
  const [status, setStatus] = useState<StudioStatus>({
    model: null,
    running: false,
    history: [],
    job: null,
    error: null,
  });
  const [text, setText] = useState('');
  const [device, setDevice] = useStoredState<NonNullable<TrainRequest['device']>>(
    'seed.training.device',
    'auto',
    isTrainingDevice,
  );
  const [corpusFileName, setCorpusFileName] = useState<string>();
  const [tab, setTab] = useState<StudioTab>('train');
  const [settings, setSettings] = useStoredState<TrainingSettings>(
    'seed.training.settings',
    DEFAULT_TRAINING_SETTINGS,
    isTrainingSettings,
  );
  const [generation, setGeneration] = useStoredState<GenerateRequest>(
    'seed.generation.settings',
    DEFAULT_GENERATION_SETTINGS,
    isGenerationSettings,
  );
  const [result, setResult] = useState<GenerationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      const revision = statusRevision.current;
      try {
        const next = await api('status');
        if (live && revision === statusRevision.current) {
          setStatus(next);
          setConnected(true);
        }
      } catch {
        if (live && revision === statusRevision.current) {
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

  async function act(fn: () => Promise<void>, rethrow = false) {
    statusRevision.current++;
    setError('');
    setNotice('');
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      if (rethrow) throw e;
    } finally {
      statusRevision.current++;
      setBusy(false);
    }
  }

  const field = (key: keyof TrainingSettings, value: string) =>
    setSettings((s) => ({
      ...s,
      [key]:
        key === 'trainingMode'
          ? value
          : ['assistantOnlyLoss', 'resetOptimizerOnModeChange', 'restartSchedule'].includes(key)
            ? value === 'true'
            : Number(value),
    }));

  const train = (resume: boolean) =>
    act(async () => {
      setStatus(
        await api('train', { text, ...settings, resume, device: resume ? undefined : device }),
      );
    });

  const loadCheckpoint = (fileName?: string) =>
    act(async () => {
      const loaded = await api('checkpoint/load', { fileName });
      setStatus(loaded);
      const loadedSettings = trainingSettingsForModel(loaded);
      if (loadedSettings) {
        setSettings(loadedSettings);
      }
      const activeDevice = loaded.backend?.device;
      if (activeDevice === 'cpu' || activeDevice === 'mps' || activeDevice === 'cuda') {
        setDevice(activeDevice);
      }
      setResult(null);
      setNotice('Model loaded. Ready to generate or resume training.');
    });

  const createCheckpoint = (name: string) =>
    act(async () => {
      const created = await api('checkpoint/create', { name, settings, device });
      setStatus(created);
      const loadedSettings = trainingSettingsForModel(created);
      if (loadedSettings) setSettings(loadedSettings);
      setResult(null);
      setNotice(`Model ${created.checkpointFileName} created and selected.`);
    }, true);

  const importCorpus = (file: File) =>
    act(async () => {
      if (file.size > 10_000_000) {
        throw new Error('File exceeds 10 MB');
      }
      setText(await file.text());
      setCorpusFileName(file.name);
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
        device={status.backend?.device}
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
            <ModelStats status={status} settings={tab === 'train' ? settings : undefined} />
            <div className="content-grid">
              <div className="primary">
                {tab === 'train' ? (
                  <CorpusEditor
                    text={text}
                    fileName={corpusFileName}
                    running={status.running}
                    busy={busy || chat.isSending}
                    onTextChange={setText}
                    onImport={importCorpus}
                    onUseSample={() => {
                      setText(SAMPLE);
                      setCorpusFileName('SAMPLE CORPUS');
                    }}
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
                  device={device}
                  onDeviceChange={setDevice}
                />
                <CheckpointControls
                  checkpoint={status.checkpoint}
                  busy={busy || chat.isSending}
                  running={status.running}
                  onLoad={loadCheckpoint}
                  onCreate={createCheckpoint}
                  fileName={status.checkpointFileName}
                  files={status.checkpoints}
                />
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
