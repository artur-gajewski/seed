export function formatDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.ceil(milliseconds / 1000));
  const parts = [Math.floor(seconds / 3600), Math.floor((seconds % 3600) / 60), seconds % 60];
  if (parts[0] === 0) parts.shift();
  if (parts[0] === 0) parts.shift();
  return parts.map((part) => String(part).padStart(2, '0')).join(':');
}
