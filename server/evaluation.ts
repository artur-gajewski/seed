import type { ChatRequest, ChatResponse } from '../shared/api.ts';

export interface EvaluationExample {
  question: string;
  answer: string;
  family: string;
}

interface Equation {
  left: number;
  right: number;
  operation: 'plus' | 'minus';
  result: number;
}

function equations(text: string): Equation[] {
  return [
    ...text.matchAll(/(-?\d+)\s*(plus|minus|[+−-])\s*(-?\d+)\s*(?:is|=|equals)\s*(-?\d+)/gi),
  ].map(([, left, operation, right, result]) => ({
    left: Number(left),
    right: Number(right),
    operation: ['plus', '+'].includes(operation.toLowerCase()) ? 'plus' : 'minus',
    result: Number(result),
  }));
}

export function parseEvaluationExamples(value: unknown): EvaluationExample[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('Evaluation dataset must be a nonempty array');
  }
  return value.map((entry: unknown, index) => {
    if (typeof entry !== 'object' || entry === null)
      throw new Error(`Invalid example ${index + 1}`);
    const { question, answer, family } = entry as Record<string, unknown>;
    if (
      typeof question !== 'string' ||
      !question.trim() ||
      question.length > 2000 ||
      typeof answer !== 'string' ||
      !answer.trim() ||
      typeof family !== 'string' ||
      !/^(add|subtract|give-receive|receive-give):\d+:\d+:\d+$/.test(family)
    )
      throw new Error(`Invalid example ${index + 1}`);
    const [, left, right, last] = family.split(':');
    const operation = family.split(':')[0];
    const expected =
      operation === 'add'
        ? Number(left) + Number(right)
        : operation === 'subtract'
          ? Number(left) - Number(right)
          : operation === 'give-receive'
            ? Number(left) - Number(right) + Number(last)
            : Number(left) + Number(right) - Number(last);
    const reference = equations(answer);
    if (
      !reference.length ||
      reference.at(-1)!.result !== expected ||
      reference.some(
        (equation) =>
          equation.result !==
          (equation.operation === 'plus'
            ? equation.left + equation.right
            : equation.left - equation.right),
      )
    )
      throw new Error(`Invalid reference arithmetic in example ${index + 1}`);
    const first = reference[0];
    const singleStep = operation === 'add' || operation === 'subtract';
    const firstOperation =
      operation === 'subtract' || operation === 'give-receive' ? 'minus' : 'plus';
    const operandsMatch =
      (first.left === Number(left) && first.right === Number(right)) ||
      (operation === 'add' && first.left === Number(right) && first.right === Number(left));
    if (
      first.operation !== firstOperation ||
      !operandsMatch ||
      (singleStep && (reference.length !== 1 || Number(last) !== 0)) ||
      (!singleStep &&
        (reference.length !== 2 ||
          reference[1].left !== first.result ||
          reference[1].right !== Number(last) ||
          reference[1].operation === firstOperation))
    )
      throw new Error(`Reference does not match arithmetic family in example ${index + 1}`);
    return { question, answer, family };
  });
}

export function scoreAnswer(example: EvaluationExample, reply: string) {
  const reference = equations(example.answer);
  const expected = reference.at(-1)!.result;
  const actual = equations(reply);
  const quantity = reply
    .trim()
    .match(/^(?:([A-Za-z]+) has|(?:the )?answer is|there are)\s+(-?\d+)\b/i);
  const leadingNumber = reply
    .trim()
    .match(
      /^(-?\d+)(?:\s+(?:are|is|remain|left|because|cookies|blocks|apples|pencils|marbles|stickers|shells|toys)\b|[.!]?\s*$)/i,
    );
  // Never treat the operand at the beginning of an equation as the final answer.
  const startsWithEquation = /^\s*-?\d+\s*(?:plus|minus|[+−-])\s*-?\d+/i.test(reply);
  const predicted = quantity
    ? Number(quantity[2])
    : startsWithEquation
      ? (actual.at(-1)?.result ?? null)
      : leadingNumber
        ? Number(leadingNumber[1])
        : null;
  const equationsCorrect = actual.length
    ? actual.every(
        (equation) =>
          equation.result ===
          (equation.operation === 'plus'
            ? equation.left + equation.right
            : equation.left - equation.right),
      )
    : null;
  const explanationMatches = actual.length
    ? actual.every((equation) =>
        reference.some(
          (expectedEquation) =>
            equation.operation === expectedEquation.operation &&
            equation.result === expectedEquation.result &&
            ((equation.left === expectedEquation.left &&
              equation.right === expectedEquation.right) ||
              (equation.operation === 'plus' &&
                equation.left === expectedEquation.right &&
                equation.right === expectedEquation.left)),
        ),
      )
    : null;
  const expectedSubject = example.answer.match(/^(\w+) has /)?.[1];
  const subjectCorrect =
    expectedSubject && quantity?.[1]
      ? quantity[1].toLowerCase() === expectedSubject.toLowerCase()
      : null;
  const numericCorrect = predicted === expected;
  return {
    expected,
    predicted,
    numericCorrect,
    equationsCorrect,
    explanationMatches,
    subjectCorrect,
    passed:
      numericCorrect &&
      equationsCorrect !== false &&
      explanationMatches !== false &&
      subjectCorrect !== false,
    exactMatch: reply.trim().replace(/\s+/g, ' ') === example.answer.trim().replace(/\s+/g, ' '),
  };
}

export interface EvaluationResult extends EvaluationExample, ReturnType<typeof scoreAnswer> {
  reply: string;
  elapsedMs: number;
  tokens: number;
  modelStep: number;
}

export async function evaluateExamples(
  examples: EvaluationExample[],
  generate: (request: ChatRequest) => ChatResponse | Promise<ChatResponse>,
  settings: Pick<ChatRequest, 'temperature' | 'topK' | 'maxTokens'>,
  onProgress: (result: EvaluationResult, completed: number) => void = () => {},
): Promise<EvaluationResult[]> {
  const results: EvaluationResult[] = [];
  for (const example of examples) {
    // Every question is an independent chat; reference answers are never sent to the model.
    const generated = await generate({
      ...settings,
      messages: [{ role: 'user', content: example.question }],
    });
    if (results.length && generated.modelStep !== results[0].modelStep) {
      throw new Error('Model changed during evaluation; results would not be comparable');
    }
    const result = { ...example, ...scoreAnswer(example, generated.reply), ...generated };
    results.push(result);
    onProgress(result, results.length);
  }
  return results;
}

export function summarizeEvaluation(results: EvaluationResult[]) {
  if (!results.length) throw new Error('No evaluation results');
  const times = results.map((result) => result.elapsedMs).sort((a, b) => a - b);
  const numericCorrect = results.filter((result) => result.numericCorrect).length;
  const passed = results.filter((result) => result.passed).length;
  return {
    total: results.length,
    numericCorrect,
    numericAccuracy: numericCorrect / results.length,
    passed,
    passRate: passed / results.length,
    exactMatches: results.filter((result) => result.exactMatch).length,
    unscored: results.filter((result) => result.predicted === null).length,
    meanResponseMs: times.reduce((sum, time) => sum + time, 0) / times.length,
    medianResponseMs:
      (times[Math.floor((times.length - 1) / 2)] + times[Math.floor(times.length / 2)]) / 2,
    p95ResponseMs: times[Math.ceil(times.length * 0.95) - 1],
    totalResponseMs: times.reduce((sum, time) => sum + time, 0),
    byOperation: Object.fromEntries(
      ['add', 'subtract', 'give-receive', 'receive-give'].map((operation) => {
        const subset = results.filter((result) => result.family.startsWith(`${operation}:`));
        return [
          operation,
          {
            total: subset.length,
            numericCorrect: subset.filter((result) => result.numericCorrect).length,
            passed: subset.filter((result) => result.passed).length,
          },
        ];
      }),
    ),
  };
}
