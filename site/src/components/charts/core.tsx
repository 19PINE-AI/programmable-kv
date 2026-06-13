import { ReactNode } from 'react'
import { ScaleContinuousNumeric } from 'd3-scale'

export const COLORS = {
  orange: '#e07b39',
  orangeSoft: '#f3c9a8',
  blue: '#3d6fb4',
  blueSoft: '#b7cbe8',
  green: '#4a8a5c',
  red: '#c0504d',
  purple: '#7b5ea7',
  gray: '#8a8a83',
  grayLight: '#c9c5b6',
  ink: '#1f1f1d',
}

export type Scale = ScaleContinuousNumeric<number, number>

export function AxisBottom({
  scale,
  y,
  ticks,
  fmt = (v: number) => String(v),
  label,
  grid,
  gridY1,
  gridY2,
}: {
  scale: Scale
  y: number
  ticks: number[]
  fmt?: (v: number) => string
  label?: string
  grid?: boolean
  gridY1?: number
  gridY2?: number
}) {
  const [x0, x1] = scale.range()
  return (
    <g>
      <line x1={x0} x2={x1} y1={y} y2={y} stroke={COLORS.grayLight} strokeWidth={1} />
      {ticks.map((t) => (
        <g key={t} transform={`translate(${scale(t)},${y})`}>
          {grid && gridY1 !== undefined && gridY2 !== undefined && (
            <line y1={gridY1 - y} y2={gridY2 - y} stroke="#eceadf" strokeWidth={1} />
          )}
          <line y2={4} stroke={COLORS.grayLight} />
          <text className="tick-label" y={16} textAnchor="middle">
            {fmt(t)}
          </text>
        </g>
      ))}
      {label && (
        <text className="axis-label" x={(x0 + x1) / 2} y={y + 32} textAnchor="middle">
          {label}
        </text>
      )}
    </g>
  )
}

export function AxisLeft({
  scale,
  x,
  ticks,
  fmt = (v: number) => String(v),
  label,
  grid,
  gridX2,
}: {
  scale: Scale
  x: number
  ticks: number[]
  fmt?: (v: number) => string
  label?: string
  grid?: boolean
  gridX2?: number
}) {
  const [y1, y0] = scale.range()
  return (
    <g>
      <line x1={x} x2={x} y1={y0} y2={y1} stroke={COLORS.grayLight} strokeWidth={1} />
      {ticks.map((t) => (
        <g key={t} transform={`translate(${x},${scale(t)})`}>
          {grid && gridX2 !== undefined && (
            <line x1={0} x2={gridX2 - x} stroke="#eceadf" strokeWidth={1} />
          )}
          <line x2={-4} stroke={COLORS.grayLight} />
          <text className="tick-label" x={-8} dy="0.32em" textAnchor="end">
            {fmt(t)}
          </text>
        </g>
      ))}
      {label && (
        <text
          className="axis-label"
          transform={`translate(${x - 42},${(y0 + y1) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {label}
        </text>
      )}
    </g>
  )
}

export function ChartSvg({
  width,
  height,
  children,
  ariaLabel,
}: {
  width: number
  height: number
  children: ReactNode
  ariaLabel?: string
}) {
  return (
    <svg
      className="chart"
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: '100%', height: 'auto', display: 'block' }}
      role="img"
      aria-label={ariaLabel}
    >
      {children}
    </svg>
  )
}

/** Legend rendered as a flex row of swatches under or above a chart. */
export function Legend({ items }: { items: { label: string; color: string; dash?: boolean }[] }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: 16,
        flexWrap: 'wrap',
        fontFamily: 'var(--sans)',
        fontSize: 11.5,
        color: 'var(--ink-soft)',
        marginTop: 6,
      }}
    >
      {items.map((it) => (
        <span key={it.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <svg width={18} height={8}>
            <line
              x1={0}
              x2={18}
              y1={4}
              y2={4}
              stroke={it.color}
              strokeWidth={2.5}
              strokeDasharray={it.dash ? '4 3' : undefined}
            />
          </svg>
          {it.label}
        </span>
      ))}
    </div>
  )
}
