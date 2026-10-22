import { mkdir, writeFile } from 'node:fs/promises';
import type { ChatMessage } from '../shared/api.ts';

// Pair-level holdouts: no template in training contains a held-out color/object pair.
const colors = ['red', 'blue', 'green', 'yellow', 'black', 'white'];
const objects = [
  'car',
  'boat',
  'ball',
  'cup',
  'hat',
  'box',
  'kite',
  'bus',
  'bike',
  'bag',
  'shoe',
  'book',
  'submarine',
  'berry',
  'blueberry',
  'train',
  'plane',
  'truck',
  'bicycle',
  'motorcycle',
  'apple',
  'banana',
  'pear',
  'grape',
  'orange',
  'lemon',
  'flower',
  'tree',
  'leaf',
  'grass',
  'shirt',
  'dress',
  'coat',
  'sock',
  'scarf',
  'glove',
  'chair',
  'table',
  'door',
  'window',
  'house',
  'roof',
  'wall',
  'fence',
  'bridge',
  'flag',
  'bottle',
  'plate',
  'bowl',
  'spoon',
  'pencil',
  'pen',
  'paper',
  'notebook',
  'phone',
  'computer',
  'screen',
  'lamp',
  'clock',
  'toy',
  'robot',
  'rocket',
  'star',
  'moon',
  'sun',
  'cloud',
  'sky',
  'ocean',
  'river',
  'fish',
  'bird',
  'cat',
  'dog',
  'horse',
  'frog',
  'butterfly',
  'beetle',
  'snake',
  'dragon',
  'umbrella',
  'balloon',
  'bucket',
  'helmet',
  'suitcase',
  'backpack',
  'skateboard',
  'fire engine',
  'sailing boat',
  'sports car',
  'rubber duck',
  'teddy bear',
  'coffee mug',
  'tennis ball',
  'school bus',
  'garden gate',
  'washing machine',
  'traffic light',
];
const unseenObjects = ['zeppelin', 'ukulele', 'axolotl', 'quadcopter', 'snowmobile', 'teapot'];
type Example = { question: string; answer: string; family: string; history?: ChatMessage[] };
const training: Example[] = [];
const evaluation: Example[] = [];
for (const [ci, color] of colors.entries()) {
  for (const [oi, object] of [...objects, ...unseenObjects].entries()) {
    const unseen = oi >= objects.length;
    const heldOut = unseen || oi % colors.length === ci;
    const destination = heldOut ? evaluation : training;
    // Choose a distractor outside its own held-out color/object pairs.
    let otherIndex = (oi + 1) % objects.length;
    while (otherIndex % colors.length === (ci + 1) % colors.length) {
      otherIndex = (otherIndex + 1) % objects.length;
    }
    const other = objects[otherIndex];
    const distractor = colors[(ci + 1) % colors.length];
    const examples: [string, string, string][] = [
      [`What color is a ${color} ${object}?`, color, 'direct'],
      [`What color is the ${color} ${object}?`, color, 'direct'],
      [`What color is ${color} ${object}?`, color, 'no-article'],
      [`What colour is a ${color} ${object}?`, color, 'british'],
      [`The ${object} is ${color}. What color is it?`, color, 'statement'],
      [`A ${color} ${object}. What is its color?`, color, 'paraphrase'],
      [`Tell me the color of the ${color} ${object}.`, color, 'instruction'],
      [`Which color does the ${color} ${object} have?`, color, 'paraphrase'],
      [`What color is a ${color} ${object}`, color, 'no-punctuation'],
      [`what color is a ${color} ${object}?`, color, 'lowercase'],
      [`What object is ${color}: a ${color} ${object}?`, object, 'object'],
      [
        `A ${color} ${object} is beside a ${distractor} ${other}. What color is the ${object}?`,
        color,
        'distractor-first',
      ],
      [
        `A ${distractor} ${other} is beside a ${color} ${object}. What color is the ${object}?`,
        color,
        'distractor-last',
      ],
    ];
    for (const [question, answer, family] of examples) {
      destination.push({ question, answer, family: unseen ? `unseen-${family}` : family });
    }
  }
}
// Teach actual chat continuations using accurate answers throughout.
// History examples come exclusively from training pairs.
const singleTurns = [...training];
for (let i = 0; i < singleTurns.length; i++) {
  const first = singleTurns[(i * 37 + 11) % singleTurns.length];
  const current = singleTurns[i];
  training.push({
    ...current,
    family: 'conversation',
    history: [
      { role: 'user', content: first.question },
      { role: 'assistant', content: `${first.answer}.` },
    ],
  });
}
const singleEvaluation = [...evaluation];
for (let i = 0; i < singleEvaluation.length; i += 3) {
  const first = singleTurns[(i * 17) % singleTurns.length];
  evaluation.push({
    ...singleEvaluation[i],
    family: 'conversation',
    history: [
      { role: 'user', content: first.question },
      { role: 'assistant', content: `${first.answer}.` },
    ],
  });
}
await mkdir(new URL('../data/inputs/', import.meta.url), { recursive: true });
await mkdir(new URL('../data/references/', import.meta.url), { recursive: true });
// Deterministic shuffle keeps the studio's contiguous validation tail representative.
let state = 20261020;
for (let i = training.length - 1; i > 0; i--) {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  const j = state % (i + 1);
  [training[i], training[j]] = [training[j], training[i]];
}
await writeFile(
  new URL('../data/inputs/color_reasoning_01.txt', import.meta.url),
  training
    .map(
      ({ question, answer, history }) =>
        `${history?.map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}${m.role === 'assistant' ? '\n\n' : '\n'}`).join('') ?? ''}User: ${question}\nAssistant: ${answer}.`,
    )
    .join('\n\n\n') + '\n',
);
await writeFile(
  new URL('../data/references/color_reasoning_01_reference.json', import.meta.url),
  JSON.stringify(evaluation, null, 2) + '\n',
);
console.log(`Generated ${training.length} training and ${evaluation.length} held-out examples.`);
