import { useState } from 'react'
import { ChartSvg } from './core'

/** Color ramp: white -> orange (or any target) through interpolation in sRGB. */
export function ramp(t: number, target = [224, 123, 57]): string {
  const c = Math.max(0, Math.min(1, t))
  const from = [251, 250, 246]
  const rgb = from.map((f, i) => Math.round(f + (target[i] - f) * c))
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
}

export function rampDiverging(t: number): string {
  // -1 (blue) .. 0 (paper) .. +1 (orange)
  if (t >= 0) return ramp(t)
  return ramp(-t, [61, 111, 180])
}

export function Heatmap({
  rows,
  cols,
  value,
  fmt = (v) => v.toFixed(2),
  width = 620,
  cell = 30,
  colorOf,
  rowLabelWidth = 130,
  colLabel,
  rowLabel,
  tooltip,
}: {
  rows: string[]
  cols: string[]
  value: (r: number, c: number) => number | null
  fmt?: (v: number) => string
  width?: number
  cell?: number
  colorOf?: (v: number | null) => string
  rowLabelWidth?: number
  colLabel?: string
  rowLabel?: string
  tooltip?: (r: number, c: number) => string | null
}) {
  const m = { l: rowLabelWidth, t: 34, r: 8, b: 8 }
  const cw = Math.min(cell * 1.6, (width - m.l - m.r) / cols.length)
  const ch = cell
  const height = m.t + rows.length * ch + m.b + (colLabel ? 14 : 0)
  const [hover, setHover] = useState<[number, number] | null>(null)
  const color = colorOf ?? ((v: number | null) => (v === null ? '#f0eee6' : ramp(v)))

  return (
    <ChartSvg width={width} height={height}>
      {colLabel && (
        <text className="axis-label" x={m.l + (cols.length * cw) / 2} y={12} textAnchor="middle">
          {colLabel}
        </text>
      )}
      {cols.map((c, j) => (
        <text
          key={c}
          className="tick-label"
          x={m.l + j * cw + cw / 2}
          y={m.t - 7}
          textAnchor="middle"
          style={{ fontSize: 10 }}
        >
          {c}
        </text>
      ))}
      {rows.map((r, i) => (
        <g key={r}>
          <text className="tick-label" x={m.l - 8} y={m.t + i * ch + ch / 2} dy="0.32em" textAnchor="end" style={{ fontSize: 11 }}>
            {r}
          </text>
          {cols.map((_, j) => {
            const v = value(i, j)
            const isH = hover?.[0] === i && hover?.[1] === j
            return (
              <g key={j}>
                <rect
                  x={m.l + j * cw + 1}
                  y={m.t + i * ch + 1}
                  width={cw - 2}
                  height={ch - 2}
                  rx={3}
                  fill={color(v)}
                  stroke={isH ? 'var(--ink)' : 'transparent'}
                  strokeWidth={1.5}
                  onMouseEnter={() => setHover([i, j])}
                  onMouseLeave={() => setHover(null)}
                >
                  {tooltip && v !== null && <title>{tooltip(i, j) ?? ''}</title>}
                </rect>
                {v !== null && cw > 34 && (
                  <text
                    className="tick-label"
                    x={m.l + j * cw + cw / 2}
                    y={m.t + i * ch + ch / 2}
                    dy="0.32em"
                    textAnchor="middle"
                    pointerEvents="none"
                    style={{ fontSize: 9.5, fill: v > 0.62 ? '#fff' : 'var(--ink-soft)', fontWeight: 500 }}
                  >
                    {fmt(v)}
                  </text>
                )}
              </g>
            )
          })}
        </g>
      ))}
      {rowLabel && (
        <text className="axis-label" transform={`translate(12,${m.t + (rows.length * ch) / 2}) rotate(-90)`} textAnchor="middle">
          {rowLabel}
        </text>
      )}
    </ChartSvg>
  )
}
