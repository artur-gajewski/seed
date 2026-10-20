import { existsSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

if (process.platform !== 'darwin') throw new Error('Apple Accelerate requires macOS.');
const root = fileURLToPath(new URL('../', import.meta.url));
const headers = [
  process.env.NODE_INCLUDE_DIR,
  join(homedir(), 'Library/Caches/node-gyp', process.versions.node, 'include/node'),
  resolve(dirname(process.execPath), '../include/node'),
  '/opt/homebrew/include/node',
  '/usr/local/include/node',
].find((directory) => directory && existsSync(join(directory, 'node_api.h')));
if (!headers)
  throw new Error(
    'Node headers not found. Set NODE_INCLUDE_DIR to a directory containing node_api.h (e.g. from node-gyp install).',
  );
mkdirSync(join(root, 'native'), { recursive: true });
const result = spawnSync(
  'xcrun',
  [
    'clang++',
    '-std=c++17',
    '-O3',
    '-DNAPI_VERSION=8',
    '-DACCELERATE_NEW_LAPACK',
    '-DNODE_GYP_MODULE_NAME=accelerate',
    '-shared',
    '-undefined',
    'dynamic_lookup',
    '-I',
    headers,
    join(root, 'native/blas.cc'),
    '-framework',
    'Accelerate',
    '-o',
    join(root, 'native/accelerate.node'),
  ],
  { stdio: 'inherit' },
);
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
console.log('Built native/accelerate.node. Restart the server to enable Accelerate.');
