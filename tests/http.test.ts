import test from 'node:test';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { isCrossOriginRequest } from '../server/http.ts';
import type { Request } from '../server/http.ts';

function request(origin?: string): Request {
  return Object.assign(Readable.from([]), {
    url: '/api/train',
    method: 'POST',
    headers: { host: '127.0.0.1:3001', origin },
  });
}

test('development accepts the Vite page origin even when the proxy rewrites Host', () => {
  const origins = ['http://localhost:5173', 'http://127.0.0.1:5173'];
  assert.equal(isCrossOriginRequest(request(origins[0]), origins), false);
  assert.equal(isCrossOriginRequest(request(origins[1]), origins), false);
  assert.equal(isCrossOriginRequest(request('https://example.com'), origins), true);
  assert.equal(isCrossOriginRequest(request('http://localhost:5173.evil.test'), origins), true);
  assert.equal(isCrossOriginRequest(request('null'), origins), true);
});

test('production only accepts same-origin requests or requests without Origin', () => {
  assert.equal(isCrossOriginRequest(request('http://127.0.0.1:3001')), false);
  assert.equal(isCrossOriginRequest(request()), false);
  assert.equal(isCrossOriginRequest(request('http://localhost:5173')), true);
});
