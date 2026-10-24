import { useEffect, useState } from 'react';
import { readPreference, writePreference } from '../lib/studioPreferences.ts';

export function useStoredState<T>(key: string, fallback: T, valid: (value: unknown) => value is T) {
  const [value, setValue] = useState<T>(() => readPreference(key, fallback, valid));
  useEffect(() => {
    writePreference(key, value, valid);
  }, [key, value, valid]);
  return [value, setValue] as const;
}
