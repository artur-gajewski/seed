export type CompositionalExample = { question: string; answer: string; family: string };

export function familyFor(question: string): string {
  const rules: [string, RegExp][] = [
    ['conditional', /^Rule:/],
    ['negation', /, not /],
    ['ordering', /ascending order/],
    ['parity', /even number/],
    ['temporal', /arrived before/],
    ['counting', /objects.*total/],
    ['multi-step-arithmetic', /coins remain/],
    ['quantity-comparison', /Who has more/],
    ['comparison', /greater than|larger than|exceed|^Compare /],
    ['multiplication', /product|[×*]|multipl|times/i],
    ['subtraction', /difference|minus|subtract|taking|Calculate \d+ -/i],
    ['addition', /sum|plus|added|^Add |Calculate \d+ \+/],
    ['state-update', /paints/],
    ['transfer', /received the/],
    ['ownership', /Who has the/],
    ['location', /Where is/],
    ['carrying', /carrying/],
    ['universal-rule', /^All /],
    ['object-selection', /^On a shelf/],
    ['color-distractor', /beside/],
    ['object', /object|Name the thing/],
    ['color', /color/i],
  ];
  return rules.find(([, pattern]) => pattern.test(question))?.[0] ?? 'unclassified';
}

export function parseCorpus(text: string): CompositionalExample[] {
  return text
    .trim()
    .split(/\r?\n\s*\r?\n/)
    .map((block, index) => {
      const match = /^User: ([^\r\n]+)\r?\nAssistant: ([^\r\n]+)$/.exec(block);
      if (!match) throw new Error(`Invalid question/answer block ${index + 1}`);
      return {
        question: match[1],
        answer: match[2]
          .trim()
          .toLowerCase()
          .replace(/[.!?]+$/, ''),
        family: familyFor(match[1]),
      };
    });
}
