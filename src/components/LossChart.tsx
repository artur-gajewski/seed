import { useEffect, useRef, useState } from 'react';
import type { LossPoint } from '../../shared/api.ts';

export function LossChart({ points }: { points: LossPoint[] }) {
  const chart = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(690);
  const hasPoints = points.length > 0;

  useEffect(() => {
    const element = chart.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setWidth(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [hasPoints]);

  if (!points.length) {
    return (
      <div className="chart-empty">
        <span className="chart-symbol">⌁</span>
        <p>No training data yet</p>
        <small>Run training to plot the loss.</small>
      </div>
    );
  }
  const values = points.flatMap((point) =>
    point.validationLoss === undefined ? [point.loss] : [point.loss, point.validationLoss],
  );
  const low = Math.max(0, Math.min(...values) - 0.2);
  const high = Math.max(...values) + 0.2;

  const right = Math.max(45, width - 26);
  const x = (i: number) => 44 + (i / Math.max(1, points.length - 1)) * (right - 44);

  const y = (v: number) => 145 - ((v - low) / (high - low)) * 125;
  const line = points.map((p, i) => `${x(i)},${y(p.loss)}`).join(' ');
  const validationLine = points
    .flatMap((point, index) =>
      point.validationLoss === undefined ? [] : [`${x(index)},${y(point.validationLoss)}`],
    )
    .join(' ');
  return (
    <svg
      ref={chart}
      className="chart"
      viewBox={`0 0 ${width} 180`}
      role="img"
      aria-label={`Training loss, from ${points[0].loss.toFixed(3)} to ${points.at(-1)!.loss.toFixed(3)}`}
    >
      <defs>
        <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--chart-line)" stopOpacity=".2" />
          <stop offset="100%" stopColor="var(--chart-line)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <line
            x1="44"
            x2={right}
            y1={20 + i * 42}
            y2={20 + i * 42}
            stroke="var(--chart-grid)"
            strokeDasharray="4 5"
          />
          <text x="0" y={24 + i * 42} fill="var(--chart-label)" fontSize="13">
            {(high - ((high - low) * i) / 3).toFixed(2)}
          </text>
        </g>
      ))}
      <polygon points={`44,148 ${line} ${x(points.length - 1)},148`} fill="url(#fill)" />
      <polyline points={line} fill="none" stroke="var(--chart-line)" strokeWidth="2.5" />
      {validationLine && (
        <polyline
          points={validationLine}
          fill="none"
          stroke="var(--text-muted)"
          strokeDasharray="5 5"
          strokeWidth="2"
        />
      )}
      <circle
        cx={x(points.length - 1)}
        cy={y(points.at(-1)!.loss)}
        r="4"
        fill="var(--chart-line)"
      />
      <text x="44" y="172" fill="var(--chart-label)" fontSize="13">
        Step {points[0].step}
      </text>
      <text x={right} y="172" textAnchor="end" fill="var(--chart-label)" fontSize="13">
        Step {points.at(-1)!.step}
      </text>
    </svg>
  );
}
