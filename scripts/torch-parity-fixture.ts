// Small, fixed cross-language fixture; never reads application checkpoints.
import { LanguageModel } from '../server/model.ts';
const model = new LanguageModel('', 8, 42, { layers: 1, heads: 2, contextSize: 8 });
const inputs = [256, 72, 105, 33];
const targets = [72, 105, 33, 257];
const initial = model.serialize();
const logits = Array.from(model.forward(inputs).values);
const gradients = model.lossAndGradients(inputs, targets);
const firstLoss = model.train(inputs, targets, 0.0003);
model.trainBatch(
  [
    { inputs, targets },
    { inputs: [3, 4, 5, 6], targets: [4, 5, 6, 7] },
  ],
  0.0002,
);
console.log(
  JSON.stringify({
    initial,
    inputs,
    targets,
    logits,
    firstLoss,
    gradients: Object.fromEntries(
      Object.entries(gradients.gradients).map(([k, v]) => [k, Array.from(v)]),
    ),
    final: model.serialize(),
    tokenization: [
      'User: Hi\nAssistant: Hello!\n\nUser: Bye\nAssistant: Bye.\n',
      'é🌱<|endofstory|>\nnext',
    ].map((text) => ({ text, tokens: model.encodeTrainingText(text) })),
  }),
);
