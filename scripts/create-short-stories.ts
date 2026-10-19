import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const names = [
  'Nina',
  'Leo',
  'Lily',
  'Ada',
  'Eli',
  'Ivy',
  'Ben',
  'Zoe',
  'Tom',
  'Max',
  'Sam',
  'Finn',
  'Ruby',
  'Ava',
  'Omar',
  'Pia',
  'Noah',
  'Mia',
  'Ella',
  'Kai',
];
const places = [
  'at the park',
  'at home',
  'at school',
  'in the garden',
  'on the beach',
  'by the pond',
  'on the hill',
  'at the farm',
  'in the yard',
  'in the woods',
  'near the lake',
  'at the playground',
];
const objects = [
  'red ball',
  'toy car',
  'blue kite',
  'small boat',
  'soft bear',
  'wooden box',
  'yellow hat',
  'pink scarf',
  'paper star',
  'green cup',
];
const colors = ['red', 'blue', 'green', 'yellow', 'pink', 'purple'];

interface Scene {
  n: string;
  m: string;
  p: string;
  o: string;
  c: string;
}
const patterns: Array<(scene: Scene) => string> = [
  ({ n, m, p, o }) =>
    `${n} found a ${o} ${p}. "Is this yours?" ${n} asked ${m}. "Yes!" said ${m}. ${n} gave it back, and ${m} smiled.`,
  ({ n, m, p }) =>
    `${n} and ${m} were ${p}. Rain began to fall. They ran under a roof and waited. When the rain stopped, they went out to play again.`,
  ({ n, m, p }) =>
    `While playing ${p}, ${n} fell and hurt a knee. ${m} went to get an adult. The adult cleaned the knee. Soon ${n} could walk again.`,
  ({ n, m, p }) =>
    `${n} brought an apple to eat ${p}. ${m} had no snack. ${n} asked an adult to cut the apple in half. They each ate a piece.`,
  ({ n, m, p, c }) =>
    `${n} planted a seed ${p}. ${m} helped water it. A tiny leaf grew. Later, they saw a ${c} flower where the seed had been.`,
  ({ n, m, p, c }) =>
    `A ${c} kite would not fly ${p}. ${n} held the string while ${m} lifted the kite. A gust of wind came. Up it went!`,
  ({ n, m, p }) =>
    `${n} built a tower ${p}. It fell when the top grew too heavy. ${m} made the bottom wider. Their next tower stayed up.`,
  ({ n, m, p }) =>
    `There was a puppy ${p}. It hid when ${n} came near. ${m} sat quietly beside ${n}. After a while, the puppy came to sniff their shoes.`,
  ({ n, m, p, c }) =>
    `${n} drew a ${c} sun ${p}. "My sun is yellow," said ${m}. "Mine is pretend," said ${n}. They made a sky with two suns.`,
  ({ n, m, p, o }) =>
    `${n} could not find a ${o} ${p}. ${m} looked under a chair. There it was! ${n} picked it up and put it away.`,
  ({ n, m, p }) =>
    `${m} heard a bird ${p}. ${n} wanted to see it too. They stood still and looked up. A little bird was singing on a branch.`,
  ({ n, m, p, c }) =>
    `${n} and ${m} sailed a ${c} toy boat ${p}. Water got inside, and it began to sink. They lifted it out and dried it before playing again.`,
  ({ n, m, p }) =>
    `${n} wanted to race ${p}, but ${m} was tired. They chose to walk instead. On the way, they spotted a snail they would have missed.`,
  ({ n, m, p }) =>
    `${n} heard a loud noise ${p} and felt scared. ${m} pointed to a truck nearby. It was tipping out stones. Now ${n} knew what made the noise.`,
  ({ n, m, p, c }) =>
    `${n} carried a ${c} cup ${p}. The cup tipped, and water spilled. ${m} brought a cloth. They wiped the floor so no one would slip.`,
  ({ n, m, p }) =>
    `${n} had muddy hands after playing ${p}. ${m} was ready for lunch. They washed with soap and water, then sat down to eat.`,
  ({ n, m, p, o }) =>
    `${m} wanted a turn with a ${o} ${p}. ${n} was still using it. They agreed to take turns. When ${n} finished, ${m} had a turn.`,
  ({ n, m, p }) =>
    `A butterfly landed near ${n} ${p}. ${m} reached toward it. "Let us just watch," said ${n}. They watched its wings until it flew away.`,
  ({ n, m, p, c }) =>
    `${n} wore a ${c} hat ${p}. The wind blew it off. ${m} caught it before it rolled away. "Thanks!" said ${n}, holding the hat tight.`,
  ({ n, m, p }) =>
    `${n} and ${m} made a cake at home after playing ${p}. They forgot the sugar. The cake tasted plain, so they ate it with sweet berries.`,
  ({ n, m, p }) =>
    `${m} could not reach a book after coming home from playing ${p}. ${n} asked an adult for help. The adult handed it down, and they read together.`,
  ({ n, m, p, c }) =>
    `${n} saw a ${c} leaf ${p}. ${m} found a different one. They put the leaves side by side and noticed that one was much bigger.`,
  ({ n, m, p }) =>
    `${n} wanted to jump over a puddle ${p}. It was too wide. ${m} found a dry path around it. They took that path and kept their feet dry.`,
  ({ n, m, p, o }) =>
    `${n} packed a ${o} for a trip. ${m} brought a snack. They spent the day ${p}. Before leaving, they checked that nothing was left behind.`,
  ({ n, m, p }) =>
    `${n} and ${m} heard thunder ${p}. An adult took them indoors. From a window, they watched the rain. They waited until the storm had passed.`,
  ({ n, m, p, c }) =>
    `After a visit to friends ${p}, ${n} wanted to send a card. ${m} helped fold some paper. ${n} drew a ${c} heart and wrote "Thank you!" inside.`,
  ({ n, m, p }) =>
    `${n} and ${m} played hide and seek ${p}. ${n} hid behind a tree. ${m} saw a shoe poking out and laughed. "I found you!"`,
  ({ n, m, p, o }) =>
    `${m} lent ${n} a ${o} ${p}. When it was time to go home, ${n} remembered to give it back. ${m} was glad to have it again.`,
  ({ n, m, p }) =>
    `${n} and ${m} found litter ${p}. They asked an adult for help. Together, they put it in a bin. The place was clean for everyone to enjoy.`,
  ({ n, m, p, c }) =>
    `${n} and ${m} made a paper plane ${p}. They colored it ${c}. At first it fell straight down. A small fold in each wing helped it glide.`,
  ({ n, m, p }) =>
    `${n} saw a long shadow ${p}. ${m} waved, and the shadow waved too. They took turns making funny shapes in the sunlight.`,
  ({ n, m, p, c }) =>
    `${n} had a ${c} ribbon ${p}. ${m} helped tie it around a gift. The knot came loose once, so they tried again. This time it held.`,
];

export function createShortStories(): Array<{ text: string; pattern: number }> {
  let state = 42;
  const pick = <T>(values: T[]): T => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return values[Math.floor((state / 4294967296) * values.length)];
  };
  const stories: Array<{ text: string; pattern: number }> = [];
  const seen = new Set<string>();
  // Give every pattern the same budget instead of letting one template dominate.
  for (let pattern = 0; pattern < patterns.length; pattern++) {
    let count = 0;
    for (let attempt = 0; count < 625 && attempt < 100_000; attempt++) {
      const n = pick(names);
      const m = pick(names.filter((name) => name !== n));
      const text = patterns[pattern]({ n, m, p: pick(places), o: pick(objects), c: pick(colors) });
      if (seen.has(text)) continue;
      assert.ok(Buffer.byteLength(text) + 2 <= 256, `Story exceeds context: ${text}`);
      seen.add(text);
      stories.push({ text, pattern });
      count++;
    }
    assert.equal(count, 625, `Insufficient unique stories for pattern ${pattern}`);
  }
  // Deterministically shuffle so the contiguous validation split includes every topic.
  for (let index = stories.length - 1; index > 0; index--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const other = Math.floor((state / 4294967296) * (index + 1));
    [stories[index], stories[other]] = [stories[other], stories[index]];
  }
  return stories;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const stories = createShortStories();
  await writeFile(
    new URL('../corpus-examples/short_stories.txt', import.meta.url),
    stories.map(({ text }) => `${text}\n<|endofstory|>`).join('\n\n') + '\n',
  );
  console.log(`Wrote ${stories.length} stories across ${patterns.length} balanced patterns.`);
}
