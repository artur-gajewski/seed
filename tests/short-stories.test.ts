import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createShortStories } from '../scripts/create-short-stories.ts';
import { LanguageModel } from '../server/model.ts';

test('short stories are reproducible, balanced, bounded, and explicitly terminated', async () => {
  const stories = createShortStories();
  const text = await readFile(
    new URL('../corpus-examples/short_stories.txt', import.meta.url),
    'utf8',
  );
  assert.equal(stories.length, 20000);
  assert.equal(new Set(stories.map(({ text }) => text)).size, 20000);
  assert.equal(text, stories.map(({ text }) => `${text}\n<|endofstory|>`).join('\n\n') + '\n');
  const counts = new Map<number, number>();
  const validationPatterns = new Set<number>();
  const model = new LanguageModel('', 8, 42, { layers: 1, heads: 2, contextSize: 256 });
  for (const [index, story] of stories.entries()) {
    counts.set(story.pattern, (counts.get(story.pattern) ?? 0) + 1);
    if (index >= 18000) validationPatterns.add(story.pattern);
    assert.ok(model.encodeTrainingText(`${story.text}\n<|endofstory|>`).length <= 256);
    assert.doesNotMatch(story.text, /\bin the (?:hill|beach)\b|\bcald\b|It did not belong to/);
  }
  assert.equal(counts.size, 32);
  assert.ok([...counts.values()].every((count) => count === 625));
  assert.equal(validationPatterns.size, 32);
});

test('story markers become stop tokens while paragraphs and dialogue retain their meaning', () => {
  const model = new LanguageModel('', 8, 42, { layers: 1, heads: 2, contextSize: 256 });
  const prose = 'First paragraph.\n\nSecond paragraph.';
  assert.deepEqual(model.encodeTrainingText(prose), [256, ...model.encode(prose)]);
  assert.deepEqual(
    model.encodeTrainingText(`${prose}<|endofstory|>\n\nNext story.<|endofstory|>`),
    [256, ...model.encode(prose), 257, ...model.encode('\n\nNext story.'), 257],
  );
  assert.deepEqual(model.encodeTrainingText('User: Hi\nAssistant: Hello\n\n'), [
    256,
    ...model.encode('User: Hi\nAssistant: Hello'),
    257,
    ...model.encode('\n\n'),
  ]);
});

test('a small model learns to finish a plain story without printing its marker', () => {
  const model = new LanguageModel('', 16, 42, { layers: 1, heads: 2, contextSize: 32 });
  const tokens = model.encodeTrainingText('Nina ran.<|endofstory|>');
  for (let step = 0; step < 500; step++) {
    model.train(tokens.slice(0, -1), tokens.slice(1), 0.003);
  }
  assert.deepEqual(model.generateDetailed('Nina', 50, 0.1, 1), {
    completion: ' ran.',
    tokens: 6,
  });
  const restored = LanguageModel.restore(model.serialize());
  assert.ok(restored instanceof LanguageModel);
  assert.deepEqual(restored.generateDetailed('Nina', 50, 0.1, 1), {
    completion: ' ran.',
    tokens: 6,
  });
});
