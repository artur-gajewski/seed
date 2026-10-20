import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import type { ChatMessage } from '../shared/api.ts';
import { LanguageModel } from '../server/model.ts';

const root = new URL('../', import.meta.url);
const read = (name: string) => readFile(new URL(name, root), 'utf8');
const base = await read('data/inputs/color_reasoning_01.txt');
interface Example {
  question: string;
  answer: string;
  family: string;
  history?: ChatMessage[];
}
const originalEvaluation = JSON.parse(
  await read('data/references/color_reasoning_01_reference.json'),
) as Example[];
const reservedPairs = new Set(
  originalEvaluation
    .filter((e) => e.family === 'direct')
    .map((e) => e.question.match(/(?:a|the) (.+)\?$/)![1]),
);
const reservedObjects = new Set(
  originalEvaluation
    .filter((e) => e.family === 'unseen-direct')
    .map((e) => e.question.match(/(?:a|the) \w+ (.+)\?$/)![1]),
);
const oldColors = ['red', 'blue', 'green', 'yellow', 'black', 'white'];
const newColors = [
  'orange',
  'purple',
  'pink',
  'brown',
  'gray',
  'grey',
  'cyan',
  'magenta',
  'silver',
  'gold',
  'turquoise',
  'navy',
  'violet',
  'beige',
  'teal',
  'maroon',
];
const colors = [...oldColors, ...newColors];
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
  'flower',
  'shirt',
  'chair',
  'door',
  'bottle',
  'pencil',
  'pen',
  'robot',
  'flag',
  'fence',
  'apple',
  'cat',
  'dog',
  'tractor',
  'scooter',
  'canoe',
  'drum',
  'guitar',
  'lantern',
  'vase',
  'cushion',
  'basket',
  'ribbon',
  'pebble',
  'shell',
  'feather',
  'apron',
  'tent',
  'sail',
  'tram',
  'wagon',
  'bench',
  'cabinet',
  'curtain',
  'rug',
  'towel',
  'soap',
  'comb',
  'brush',
  'button',
  'bead',
  'gem',
  'crayon',
  'marker',
  'envelope',
  'parcel',
  'mailbox',
  'watering can',
  'toy dinosaur',
  'rain boot',
  'lunch box',
];
const originalPairReserved = (pair: string) =>
  [...reservedPairs].some((reserved) => pair === reserved || pair.startsWith(`${reserved} `));
const heldOutPairs = new Set<string>();
for (const [ci, color] of colors.entries()) {
  for (const [oi, object] of objects.entries()) {
    const pair = `${color} ${object}`;
    // New evaluation pairs must also be absent from the earlier training corpus.
    if ((ci + oi) % 11 === 0 && !base.includes(pair) && !originalPairReserved(pair)) {
      heldOutPairs.add(pair);
    }
  }
}
const allowed = (color: string, object: string) =>
  !reservedObjects.has(object) &&
  !originalPairReserved(`${color} ${object}`) &&
  !heldOutPairs.has(`${color} ${object}`);
const model = new LanguageModel('', 16, 42, { layers: 1, heads: 1, contextSize: 256 });
const blocks = new Set<string>();
const lessons: Example[] = [];
const evaluation: Example[] = [];
const render = (messages: ChatMessage[]) =>
  messages
    .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
    .join('\n')
    .replace(/\nUser:/g, '\n\nUser:');
const add = (messages: ChatMessage[]) => {
  const block = render(messages);
  assert.ok(model.encodeTrainingText(block).length <= 256, 'Lesson exceeds model context');
  blocks.add(block);
};
const turns = (examples: Example[]): ChatMessage[] =>
  examples.flatMap((e) => [
    { role: 'user' as const, content: e.question },
    { role: 'assistant' as const, content: `${e.answer}.` },
  ]);
for (const [ci, color] of colors.entries()) {
  for (const [oi, object] of objects.entries()) {
    if (originalPairReserved(`${color} ${object}`)) continue;
    const examples: Example[] = [
      { question: `What color is a ${color} ${object}?`, answer: color, family: 'direct' },
      { question: `What color is ${color} ${object}?`, answer: color, family: 'no-article' },
      { question: `What color is a ${color} ${object}`, answer: color, family: 'no-punctuation' },
      { question: `What colour is the ${color} ${object}?`, answer: color, family: 'british' },
      {
        question: `Please name the color of this ${color} ${object}.`,
        answer: color,
        family: 'wording',
      },
      { question: `I see a ${color} ${object}. Its color?`, answer: color, family: 'wording' },
      { question: `Which hue is the ${color} ${object}?`, answer: color, family: 'wording' },
      {
        question: `The ${object} is painted ${color}. Name its color.`,
        answer: color,
        family: 'statement',
      },
      {
        question: `Answer with a color: ${color} ${object}.`,
        answer: color,
        family: 'instruction',
      },
      {
        question: `Can you tell me the colour of a ${color} ${object}?`,
        answer: color,
        family: 'wording',
      },
      { question: `what color is the ${color} ${object}`, answer: color, family: 'lowercase' },
      { question: `Name the object in "${color} ${object}".`, answer: object, family: 'object' },
      {
        question: `What object is ${color}: a ${color} ${object}?`,
        answer: object,
        family: 'object',
      },
    ];
    if (heldOutPairs.has(`${color} ${object}`)) {
      evaluation.push(...examples);
      continue;
    }
    assert.ok(allowed(color, object));
    for (const example of examples) {
      lessons.push(example);
      add(turns([example]));
    }
    let otherIndex = (oi + 1) % objects.length;
    const otherColor = colors[(ci + 5) % colors.length];
    while (otherIndex === oi || !allowed(otherColor, objects[otherIndex])) {
      otherIndex = (otherIndex + 1) % objects.length;
    }
    const other = objects[otherIndex];
    for (const scene of [
      `A ${color} ${object} and a ${otherColor} ${other}.`,
      `A ${otherColor} ${other} and a ${color} ${object}.`,
    ]) {
      add(
        turns([
          {
            question: `${scene} What color is the ${object}?`,
            answer: color,
            family: 'distractor',
          },
        ]),
      );
    }
    // Follow-ups refer to the scene, then change one stated property explicitly.
    add(
      turns([
        { question: `A ${color} ${object}. What color is it?`, answer: color, family: 'context' },
        { question: 'Repeat its color.', answer: color, family: 'context' },
        {
          question: `Paint it ${otherColor}. Its color now?`,
          answer: otherColor,
          family: 'update',
        },
        { question: 'And now, what color is it?', answer: otherColor, family: 'context' },
      ]),
    );
  }
}
// Interleave independent questions to teach switching subjects over 3–5 turns.
for (let i = 0; i < lessons.length; i += 3) {
  const session: Example[] = [];
  for (let turn = 0; turn < 3 + (Math.floor(i / 3) % 3); turn++) {
    const candidate = lessons[(i + turn * 137) % lessons.length];
    if (model.encodeTrainingText(render(turns([...session, candidate]))).length > 256) break;
    session.push(candidate);
  }
  if (session.length >= 3) add(turns(session));
}
// Compact five-turn histories fit even the existing 256-byte context.
for (let i = 0; i < 1000; i++) {
  const session: Example[] = [];
  for (let turn = 0; turn < 5; turn++) {
    const color = colors[(i + turn * 3) % colors.length];
    let oi = (i * 7 + turn * 11) % objects.length;
    while (!allowed(color, objects[oi]) || objects[oi].length > 5) oi = (oi + 1) % objects.length;
    session.push({
      question: `${color} ${objects[oi]}. Color?`,
      answer: color,
      family: 'compact-chat',
    });
  }
  if (model.encodeTrainingText(render(turns(session))).length <= 256) add(turns(session));
}
// Replay a deterministic sample of old lessons. Keep all old evaluation holdouts intact.
let replay = 0;
for (const [i, block] of base.trim().split('\n\n').entries()) {
  if (i % 3 !== 0) continue;
  const normalized = block.replace(/\nUser:/g, '\n\nUser:');
  if (model.encodeTrainingText(normalized).length > 256) continue;
  blocks.add(normalized);
  replay++;
}
// Reserve a wording absent from both training corpora, with and without chat history.
for (const pair of heldOutPairs) {
  const [color, ...words] = pair.split(' ');
  const object = words.join(' ');
  const otherColor = colors[(colors.indexOf(color) + 5) % colors.length];
  const other = objects[(objects.indexOf(object) + 1) % objects.length];
  for (const scene of [
    `A ${color} ${object} and a ${otherColor} ${other}.`,
    `A ${otherColor} ${other} and a ${color} ${object}.`,
  ]) {
    evaluation.push({
      question: `${scene} What color is the ${object}?`,
      answer: color,
      family: 'distractor',
    });
  }
  const updateHistory = turns([
    { question: `A ${color} ${object}. What color is it?`, answer: color, family: 'context' },
    { question: `Paint it ${otherColor}. Its color now?`, answer: otherColor, family: 'update' },
  ]);
  const updateExample = {
    question: 'Remind me of its current color.',
    answer: otherColor,
    family: 'color-update-history',
    history: updateHistory,
  };
  assert.ok(
    model.encodeTrainingText(render([...updateHistory, ...turns([updateExample])])).length <= 256,
  );
  evaluation.push(updateExample);
  const question = `Identify the color here: a ${color} ${object}.`;
  evaluation.push({ question, answer: color, family: 'unseen-wording' });
  const history = turns([lessons[17], lessons[431], lessons[817]]);
  const conversation = { question, answer: color, family: 'long-conversation', history };
  if (model.encodeTrainingText(render([...history, ...turns([conversation])])).length <= 256) {
    evaluation.push(conversation);
  }
}
let state = 20261022;
const output = [...blocks];
for (let i = output.length - 1; i > 0; i--) {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  const j = state % (i + 1);
  [output[i], output[j]] = [output[j], output[i]];
}
const corpus = output.join('\n\n\n') + '\n';
for (const example of [...originalEvaluation, ...evaluation]) {
  assert.ok(
    !corpus.includes(`User: ${example.question}\n`),
    `Evaluation question leaked: ${example.question}`,
  );
}
for (const pair of [...reservedPairs, ...heldOutPairs]) {
  assert.ok(!new RegExp(`\\b${pair}\\b`).test(corpus), `Evaluation pair leaked: ${pair}`);
}
await writeFile(new URL('data/inputs/color_reasoning_02.txt', root), corpus);
await writeFile(
  new URL('data/references/color_reasoning_02_reference.json', root),
  JSON.stringify(evaluation, null, 2) + '\n',
);
console.log(
  JSON.stringify({
    blocks: output.length,
    replay,
    evaluation: evaluation.length,
    colors: colors.length,
    bytes: Buffer.byteLength(corpus),
  }),
);
