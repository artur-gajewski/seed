// Recognize explicit correction cues, rather than asking the untrained model to classify them.
const correctionPrefixes = [
  /^(?:no|nope|nah|not quite|wrong|incorrect|actually|correction|instead)\s*[.!,:;—–-]+\s*/i,
  /^(?:that['’]s|that is|you['’]re|you are) (?:wrong|incorrect|not (?:right|correct))\s*[.!,:;—–-]+\s*/i,
  /^(?:the )?(?:correct|right) answer\s+(?:is\b\s*[:—–-]?\s*|[:—–-]\s*)/i,
  /^(?:the )?(?:correct|right) answer\s*[:—–-]\s*/i,
  /^(?:the )?answer should be\b\s*[:—–-]?\s*/i,
  /^(?:it should be|you should say|instead say|i meant)\b\s*[:—–-]?\s*/i,
];
const questionStart =
  /^(?:can|could|would|will|do|does|did|is|are|was|were|what|why|when|where|who|how|should)\b/i;
const feedbackOnly =
  /^(?:no|nope|nah|wrong|incorrect|not quite|(?:that['’]s|that is|you['’]re|you are) (?:wrong|incorrect|not (?:right|correct))|(?:please )?try again)[.!,:;\s]*$/i;

export function newLifeInputContent(draft: string) {
  const text = draft.trim();
  if (text.startsWith('!')) return { isTeaching: true, content: text.slice(1).trim() };
  let content = text;
  for (let pass = 0; pass < 4; pass++) {
    const prefix = correctionPrefixes.find((pattern) => pattern.test(content));
    if (!prefix) break;
    const stripped = content.replace(prefix, '').trim();
    if (stripped === content || !stripped) break;
    content = stripped;
  }
  // A correction must supply an answer. Leave questions and bare feedback as normal messages.
  if (
    content === text ||
    content.endsWith('?') ||
    questionStart.test(content) ||
    feedbackOnly.test(content)
  ) {
    return { isTeaching: false, content: text };
  }
  return { isTeaching: true, content };
}

export function parseNewLifeInput(draft: string, latestMessage?: string) {
  const { isTeaching, content } = newLifeInputContent(draft);
  if (!content || new TextEncoder().encode(content).length > 100) {
    throw new Error('Enter 1 to 100 UTF-8 bytes. Teaching prefixes do not count.');
  }
  if (isTeaching) {
    if (!latestMessage) throw new Error('Send a message first, then teach an answer to its reply.');
    return { kind: 'teach', message: latestMessage, answer: content } as const;
  }
  return { kind: 'message', message: content } as const;
}
