import { cp, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

// Instrument an isolated copy so production kernels incur no timer or profiler overhead.
const directory = await mkdtemp(join(tmpdir(), 'seed-profile-'));
await cp(new URL('../server/', import.meta.url), join(directory, 'server'), { recursive: true });
try {
  await cp(new URL('../native/', import.meta.url), join(directory, 'native'), { recursive: true });
} catch (error) {
  if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
}
const tensorPath = join(directory, 'server/tensor.ts');
let source = await readFile(tensorPath, 'utf8');
source += '\nexport const operationTimes = new Map<string, number>();\n';
for (const [name, tapeIndex] of [
  ['linear', 3],
  ['causalAttention', 2],
  ['layerNorm', 3],
  ['gelu', 1],
  ['add', 2],
  ['crossEntropy', -1],
] as const) {
  source = source.replace(`export function ${name}(`, `function ${name}Implementation(`);
  source += `
export function ${name}(...args: Parameters<typeof ${name}Implementation>) {
  const tape = ${tapeIndex < 0 ? 'null' : `args[${tapeIndex}]`};
  const first = tape?.length ?? 0;
  const started = performance.now();
  const result = ${name}Implementation(...args);
  operationTimes.set('${name}.forward', (operationTimes.get('${name}.forward') ?? 0) + performance.now() - started);
  if (tape) for(let i = first; i < tape.length; i++) {
    const operation = tape[i];
    tape[i] = () => {
      const started = performance.now(); operation();
      operationTimes.set('${name}.backward', (operationTimes.get('${name}.backward') ?? 0) + performance.now() - started);
    };
  }
  return result;
}
`;
}
await writeFile(tensorPath, source);
const runner = `
import {LanguageModel} from './server/model.ts';
import {operationTimes} from './server/tensor.ts';
import {tensorBackend} from './server/numerics.ts';
const m=new LanguageModel('',256,42,{layers:4,heads:8,contextSize:256});
const batch=Array.from({length:8},(_,s)=>({inputs:Array.from({length:256},(_,i)=>(i+s*17)%256), targets:Array.from({length:256},(_,i)=>(i+s*17+1)%256)}));
let optimizerMs=0;
const optimizer=m.applyAdamW.bind(m);
m.applyAdamW=(rate)=>{const start=performance.now(); optimizer(rate); optimizerMs+=performance.now()-start;};
m.trainBatch(batch,.0003);
operationTimes.clear(); optimizerMs=0;
const start=performance.now(); const loss=m.trainBatch(batch,.0003); const totalMs=performance.now()-start;
const operations=Object.fromEntries([...operationTimes].sort((a,b)=>b[1]-a[1]));
console.log(JSON.stringify({backend:tensorBackend,totalMs,optimizerMs,loss,operations},null,2));
`;
await writeFile(join(directory, 'run.mjs'), runner);
const result = spawnSync(process.execPath, [join(directory, 'run.mjs')], {
  stdio: 'inherit',
  env: process.env,
});
if (result.error) throw result.error;
if (result.signal) throw new Error(`Profiler stopped by ${result.signal}`);
process.exitCode = result.status ?? 1;
