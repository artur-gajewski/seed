import { spawn } from 'node:child_process';

const children = [
  spawn(process.execPath, ['server/index.ts'], {
    stdio: 'inherit',
    env: { ...process.env, NODE_ENV: 'development' },
  }),
  spawn(process.execPath, ['node_modules/vite/bin/vite.js'], { stdio: 'inherit' }),
];
let exiting = false;

function stop(code = 0) {
  if (exiting) {
    return;
  }
  exiting = true;
  children.forEach((child) => child.kill());
  process.exitCode = code;
}

children.forEach((child) => {
  child.on('error', (error) => {
    console.error(error);
    stop(1);
  });
  child.on('exit', (code) => stop(code ?? 0));
});
process.on('SIGINT', () => stop());
process.on('SIGTERM', () => stop());
