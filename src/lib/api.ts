import type { ApiRequests, ApiResponses, ApiRoute } from '../../shared/api.ts';

export async function api<R extends ApiRoute>(
  route: R,
  ...args: ApiRequests[R] extends undefined ? [] : [body: ApiRequests[R]]
): Promise<ApiResponses[R]> {
  const body = args[0];
  const response = await fetch(
    `/api/${route}`,
    body === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
  );
  const data: unknown = await response.json();
  if (!response.ok) {
    const message =
      typeof data === 'object' && data !== null && 'error' in data && typeof data.error === 'string'
        ? data.error
        : 'Request failed';
    throw new Error(message);
  }
  return data as ApiResponses[R];
}
