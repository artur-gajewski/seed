import { existsSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

if (process.platform !== 'linux') throw new Error('The OpenBLAS backend requires Linux.');
const root = fileURLToPath(new URL('../', import.meta.url));
const headers = [
  process.env.NODE_INCLUDE_DIR,
  join(homedir(), '.cache/node-gyp', process.versions.node, 'include/node'),
  resolve(dirname(process.execPath), '../include/node'),
  '/usr/include/node',
  '/usr/local/include/node',
].find((directory) => directory && existsSync(join(directory, 'node_api.h')));
if (!headers)
  throw new Error(
    'Node headers not found. Set NODE_INCLUDE_DIR to the matching Node 24 include/node directory.',
  );
const pkg = spawnSync('pkg-config', ['--cflags', '--libs', 'openblas'], { encoding: 'utf8' });
if (pkg.error || pkg.status !== 0)
  throw new Error(
    'OpenBLAS development files not found. Install g++, pkg-config and libopenblas-dev. ' +
      (pkg.stderr ?? ''),
  );
mkdirSync(join(root, 'native'), { recursive: true });
const result = spawnSync(
  process.env.CXX ?? 'c++',
  [
    '-std=c++17',
    '-O3',
    '-fPIC',
    '-shared',
    '-DNAPI_VERSION=8',
    '-DSEED_OPENBLAS',
    '-DNODE_GYP_MODULE_NAME=openblas',
    '-I',
    headers,
    join(root, 'native/blas.cc'),
    '-o',
    join(root, 'native/openblas.node'),
    ...pkg.stdout.trim().split(/\s+/).filter(Boolean),
  ],
  { stdio: 'inherit' },
);
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
console.log('Built native/openblas.node. Restart the server to enable OpenBLAS.');
