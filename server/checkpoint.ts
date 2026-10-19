import { serialize, deserialize } from 'node:v8';

// Binary serialization avoids V8's single-string limit for large weight/Adam snapshots.
// Keep reading the original JSON format so existing checkpoints remain usable.
export function encodeCheckpoint(checkpoint: unknown): Buffer {
  return serialize(checkpoint);
}

export function decodeCheckpoint(contents: Buffer): unknown {
  return contents[0] === 0xff ? deserialize(contents) : JSON.parse(contents.toString('utf8'));
}
