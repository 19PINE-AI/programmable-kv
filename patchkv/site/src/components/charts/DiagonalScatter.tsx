import { scaleLinear } from 'd3-scale'
import { AxisBottom, AxisLeft, ChartSvg, COLORS } from './core'

export function DiagonalScatter({
  points,
  width = 420,
  height = 420,
  domain,
  xLabel,
  yLabel,
}: {
  points: { x: number; y: number; label: string; color: string; symbol?: 'circle' | 'square' }[]
  width?: number
  height?: number
  domain?: [number, number]
  xLabel?: string
  yLabel?: string
}) {
  const m = { l: 56, r: 14, t: 14, b: 48 }
  const vals = points.flatMap((p) => [p.x, p.y])
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const pad = (hi - lo) * 0.12 + 0.02
  const d = domain ?? [lo - pad, hi + pad]
  const x = scaleLinear().domain(d).range([m.l, width - m.r])
  const y = scaleLinear().domain(d).range([height - m.b, m.t])

  return (
    <ChartSvg width={width} height={height}>
      <AxisBottom scale={x} y={height - m.b} ticks={x.ticks(5)} label={xLabel} grid gridY1={m.t} gridY2={height - m.b} fmt={(v) => v.toFixed(1)} />
      <AxisLeft scale={y} x={m.l} ticks={y.ticks(5)} label={yLabel} grid gridX2={width - m.r} fmt={(v) => v.toFixed(1)} />
      {/* y = x diagonal */}
      <line x1={x(d[0])} y1={y(d[0])} x2={x(d[1])} y2={y(d[1])} stroke={COLORS.gray} strokeDasharray="4 4" strokeWidth={1.2} />
      <text className="tick-label" x={x(d[1]) - 4} y={y(d[1]) + 14} textAnchor="end" style={{ fontStyle: 'italic' }}>
        composed = recomputed
      </text>
      {points.map((p, i) => (
        <g key={i}>
          {p.symbol === 'square' ? (
            <rect x={x(p.x) - 5} y={y(p.y) - 5} width={10} height={10} fill={p.color} stroke="#fff" strokeWidth={1.2} rx={1.5}>
              <title>{p.label}</title>
            </rect>
          ) : (
            <circle cx={x(p.x)} cy={y(p.y)} r={6} fill={p.color} stroke="#fff" strokeWidth={1.2}>
              <title>{p.label}</title>
            </circle>
          )}
        </g>
      ))}
    </ChartSvg>
  )
}
