export function parseNewLifeInput(draft: string, latestMessage?: string) {
  const text = draft.trim();
  const isTeaching = text.startsWith('!');
  const content = isTeaching ? text.slice(1).trim() : text;
  if (!content || new TextEncoder().encode(content).length > 100) {
    throw new Error('Enter 1 to 100 UTF-8 bytes. The ! teaching prefix does not count.');
  }
  if (isTeaching) {
    if (!latestMessage) throw new Error('Send a message first, then use ! to teach its answer.');
    return { kind: 'teach', message: latestMessage, answer: content } as const;
  }
  return { kind: 'message', message: content } as const;
}
