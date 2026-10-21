import { readFile, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { decodeCheckpoint, encodeCheckpoint } from '../server/checkpoint.ts';
import { LanguageModel } from '../server/model.ts';
import type { ModelCheckpoint } from '../server/model.ts';

const { values } = parseArgs({
  options: {
    input: { type: 'string' },
    output: { type: 'string' },
    'to-seed': { type: 'boolean', default: false },
  },
});
if (!values.input || !values.output || values.input === values.output) {
  throw new Error('Supply distinct --input PATH and --output PATH');
}
const decoded = decodeCheckpoint(await readFile(values.input)) as {
  model: ModelCheckpoint;
};
const model = LanguageModel.restore(decoded.model);
if (!(model instanceof LanguageModel))
  throw new Error('Only Transformer v2 checkpoints are supported');
const checkpoint = { ...decoded, model: model.serialize() };
await writeFile(
  values.output,
  values['to-seed'] ? encodeCheckpoint(checkpoint) : JSON.stringify(checkpoint),
  { flag: 'wx' },
);
