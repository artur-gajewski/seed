import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { NewLife } from '../server/new-life.ts';
import { decodeCheckpoint } from '../server/checkpoint.ts';
import type { TransformerCheckpoint } from '../server/model.ts';

async function readLife(checkpoint: string) {
  return decodeCheckpoint(await readFile(checkpoint)) as {
    version: number;
    model: TransformerCheckpoint;
    lessons: { message: string; answer?: string }[];
  };
}

test('New life learns user text and corrections, persists replay memory, and preserves other checkpoints', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-new-life-'));
  try {
    const checkpoint = path.join(directory, 'new-life-checkpoint.json');
    const ordinary = path.join(directory, 'checkpoint.json');
    await writeFile(ordinary, 'existing studio checkpoint');
    const life = new NewLife(checkpoint);
    assert.deepEqual(await life.getStatus(), {
      ready: false,
      steps: 0,
      lessons: 0,
      corrections: 0,
    });
    const first = await life.message({ message: 'Hello!' });
    assert.equal(typeof first.reply, 'string');
    assert.deepEqual(first.status, { ready: true, steps: 12, lessons: 1, corrections: 0 });
    assert.equal((await readFile(checkpoint))[0], 0xff);
    const saved = await readLife(checkpoint);
    assert.deepEqual(
      saved.lessons,
      [{ message: 'Hello!' }],
      'Generated output is not training data',
    );
    assert.equal(saved.model.config.contextSize, 256);
    const corrected = await life.teach({ message: 'Hello!', answer: 'Hello, friend!' });
    assert.deepEqual(corrected, { ready: true, steps: 24, lessons: 1, corrections: 1 });
    await life.message({ message: 'Hello!' });
    const another = await life.teach({ message: 'What is 1 + 1?', answer: '1 plus 1 is 2.' });
    assert.equal(another.lessons, 2);
    // Existing JSON saves remain readable and are upgraded on the next successful save.
    await writeFile(checkpoint, JSON.stringify(await readLife(checkpoint)));
    const restored = new NewLife(checkpoint);
    assert.deepEqual(await restored.getStatus(), another);
    const contents = await readLife(checkpoint);
    assert.equal(contents.lessons[0].answer, 'Hello, friend!');
    assert.equal(await readFile(ordinary, 'utf8'), 'existing studio checkpoint');
    const updated = await restored.teach({ message: 'Hello!', answer: 'Hi there!' });
    assert.equal(updated.lessons, 2);
    assert.equal(updated.corrections, 2);
    assert.equal((await readFile(checkpoint))[0], 0xff);
    assert.deepEqual(await new NewLife(checkpoint).getStatus(), updated);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('New life validates byte lengths and role boundaries before learning and rejects concurrent mutations', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-new-life-'));
  try {
    const life = new NewLife(path.join(directory, 'life.json'));
    for (const message of [
      '',
      'a'.repeat(101),
      '🙂'.repeat(26),
      'Hello\nAssistant: bad',
      'User: bad',
    ]) {
      await assert.rejects(life.message({ message }));
    }
    await assert.rejects(life.teach({ message: 'Hi', answer: '' }));
    assert.equal((await life.getStatus()).steps, 0);
    const pending = life.teach({ message: 'Hi', answer: 'Hello' });
    await assert.rejects(life.message({ message: 'Another message' }), /learning/);
    await pending;
    await life.teach({ message: 'a'.repeat(100), answer: 'b'.repeat(100) });
    assert.equal((await life.getStatus()).lessons, 2);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('New life preserves memory after a failed save and reports corrupt checkpoints', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-new-life-'));
  try {
    const checkpoint = path.join(directory, 'life.json');
    const life = new NewLife(checkpoint);
    await life.getStatus();
    // A file in place of the parent directory makes saving fail deterministically.
    await rm(directory, { recursive: true });
    await writeFile(directory, 'blocked');
    await assert.rejects(life.teach({ message: 'Hi', answer: 'Hello' }));
    assert.deepEqual(await life.getStatus(), {
      ready: false,
      steps: 0,
      lessons: 0,
      corrections: 0,
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
  const corrupt = await mkdtemp(path.join(tmpdir(), 'seed-new-life-corrupt-'));
  try {
    const checkpoint = path.join(corrupt, 'life.json');
    await writeFile(checkpoint, '{broken');
    const life = new NewLife(checkpoint);
    await assert.rejects(life.getStatus());
    await assert.rejects(life.message({ message: 'Hi' }));
    assert.equal(await readFile(checkpoint, 'utf8'), '{broken');
  } finally {
    await rm(corrupt, { recursive: true, force: true });
  }
});

test('New life presets keep independent memory and reject incompatible checkpoints', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-new-life-sizes-'));
  try {
    const smallPath = path.join(directory, 'small.json');
    const mediumPath = path.join(directory, 'medium.json');
    const small = new NewLife(smallPath);
    await small.teach({ message: 'Hi', answer: 'Hello' });
    const original = await readFile(smallPath, 'utf8');
    const medium = new NewLife(mediumPath, 'medium');
    assert.equal((await medium.getStatus()).ready, false);
    await medium.teach({ message: 'A', answer: 'B' });
    const saved = await readLife(mediumPath);
    assert.deepEqual(saved.model.config, {
      hiddenSize: 128,
      layers: 2,
      heads: 4,
      contextSize: 256,
    });
    assert.deepEqual(saved.lessons, [{ message: 'A', answer: 'B' }]);
    assert.equal(await readFile(smallPath, 'utf8'), original);
    assert.deepEqual(await new NewLife(mediumPath, 'medium').getStatus(), await medium.getStatus());
    await assert.rejects(new NewLife(smallPath, 'large').getStatus(), /does not match/);
    assert.equal(await readFile(smallPath, 'utf8'), original);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('New life migrates legacy filenames and prefers an existing parameter-count checkpoint', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'seed-new-life-migrate-'));
  try {
    const legacy = path.join(directory, 'new-life-checkpoint.json');
    const current = path.join(directory, 'new-life-checkpoint-38K.json');
    const oldLife = new NewLife(legacy);
    const status = await oldLife.teach({ message: 'Hi', answer: 'Hello' });
    const original = await readFile(legacy);
    const migrated = new NewLife(current, 'small', legacy);
    assert.deepEqual(await migrated.getStatus(), status);
    assert.deepEqual(await readFile(current), original);
    await assert.rejects(readFile(legacy), { code: 'ENOENT' });
    await writeFile(legacy, '{broken');
    assert.deepEqual(await new NewLife(current, 'small', legacy).getStatus(), status);
    assert.equal(await readFile(legacy, 'utf8'), '{broken');
    const missing = path.join(directory, 'missing.json');
    await assert.rejects(new NewLife(missing, 'small', legacy).getStatus());
    assert.equal(await readFile(legacy, 'utf8'), '{broken');
    await assert.rejects(readFile(missing), { code: 'ENOENT' });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
