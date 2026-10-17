export class TrainingTiming {
  private durations: number[] = [];
  private lastUpdateAt: number;

  constructor(lastUpdateAt: number = performance.now()) {
    this.lastUpdateAt = lastUpdateAt;
  }

  recordUpdate(remainingUpdates: number, now = performance.now()): number | undefined {
    this.durations.push(now - this.lastUpdateAt);
    this.lastUpdateAt = now;
    if (this.durations.length > 10) this.durations.shift();
    if (remainingUpdates === 0) return 0;
    if (this.durations.length < 10) return undefined;
    return (this.durations.reduce((sum, duration) => sum + duration, 0) / 10) * remainingUpdates;
  }
}
