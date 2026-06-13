import { useMemo, useState } from 'react'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line as d3line, area as d3area, curveMonotoneX } from 'd3-shape'
import { AxisBottom, AxisLeft, ChartSvg, COLORS } from './core'

export interface Pt {
  x: number
  y: number
  lo?: number
  hi?: number
}

export interface Series {
  id: string
  label?: string
  color: string
  points: Pt[]
  dash?: boolean
  band?: boolean // draw lo/hi as a translucent band
  marker?: boolean
}

export function LineChart({
  series,
  height = 280,
  width = 620,
  xLabel,
  yLabel,
  xLog = false,
  yLog = false,
  yDomain,
  xDomain,
  xTicks,
  yTicks,
  xFmt,
  yFmt,
  refLinesY = [],
  highlightX,
  onHoverX,
}: {
  series: Series[]
  height?: number
  width?: number
  xLabel?: string
  yLabel?: string
  xLog?: boolean
  yLog?: boolean
  yDomain?: [number, number]
  xDomain?: [number, number]
  xTicks?: number[]
  yTicks?: number[]
  xFmt?: (v: number) => string
  yFmt?: (v: number) => string
  refLinesY?: { y: number; label?: string; color?: string }[]
  highlightX?: number | null
  onHoverX?: (x: number | null) => void
}) {
  const m = { l: 56, r: 16, t: 12, b: 44 }
  const all = series.flatMap((s) => s.points)
  const xs = all.map((p) => p.x)
  const ys = all.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y])

  const xd = xDomain ?? [Math.min(...xs), Math.max(...xs)]
  const yd = yDomain ?? [Math.min(0, ...ys), Math.max(...ys) * 1.05]

  const xScale = (xLog ? scaleLog() : scaleLinear()).domain(xd).range([m.l, width - m.r])
  const yScale = (yLog ? scaleLog() : scaleLinear()).domain(yd).range([height - m.b, m.t])

  const xt = xTicks ?? xScale.ticks(6)
  const yt = yTicks ?? yScale.ticks(5)

  const mkLine = useMemo(
    () =>
      d3line<Pt>()
        .x((p) => xScale(p.x))
        .y((p) => yScale(p.y))
        .curve(curveMonotoneX),
    [series, width, height, xLog, yLog, yDomain?.[0], yDomain?.[1]],
  )
  const mkBand = d3area<Pt>()
    .x((p) => xScale(p.x))
    .y0((p) => yScale(p.lo ?? p.y))
    .y1((p) => yScale(p.hi ?? p.y))
    .curve(curveMonotoneX)

  const [hover, setHover] = useState<{ sx: number; pts: { s: Series; p: Pt }[] } | null>(null)

  function handleMove(e: React.MouseEvent<SVGRectElement>) {
    const rect = (e.target as SVGRectElement).getBoundingClientRect()
    const fx = m.l + ((e.clientX - rect.left) / rect.width) * (width - m.l - m.r)
    const xv = xScale.invert(fx)
    const pts = series
      .map((s) => {
        const p = s.points.reduce((a, b) =>
          Math.abs(xScale(b.x) - fx) < Math.abs(xScale(a.x) - fx) ? b : a,
        )
        return { s, p }
      })
      .filter(Boolean)
    setHover({ sx: pts[0] ? xScale(pts[0].p.x) : fx, pts })
    onHoverX?.(pts[0]?.p.x ?? null)
  }

  return (
    <ChartSvg width={width} height={height}>
      <AxisBottom
        scale={xScale}
        y={height - m.b}
        ticks={xt}
        fmt={xFmt}
        label={xLabel}
        grid
        gridY1={m.t}
        gridY2={height - m.b}
      />
      <AxisLeft scale={yScale} x={m.l} ticks={yt} fmt={yFmt} label={yLabel} grid gridX2={width - m.r} />

      {refLinesY.map((r, i) => (
        <g key={i}>
          <line
            x1={m.l}
            x2={width - m.r}
            y1={yScale(r.y)}
            y2={yScale(r.y)}
            stroke={r.color ?? COLORS.gray}
            strokeDasharray="3 4"
            strokeWidth={1}
          />
          {r.label && (
            <text className="tick-label" x={width - m.r} y={yScale(r.y) - 5} textAnchor="end">
              {r.label}
            </text>
          )}
        </g>
      ))}

      {series.map((s) => (
        <g key={s.id}>
          {s.band && <path d={mkBand(s.points) ?? undefined} fill={s.color} opacity={0.13} />}
          <path
            d={mkLine(s.points) ?? undefined}
            fill="none"
            stroke={s.color}
            strokeWidth={2.2}
            strokeDasharray={s.dash ? '5 4' : undefined}
          />
          {s.marker !== false &&
            s.points.map((p, i) => (
              <circle key={i} cx={xScale(p.x)} cy={yScale(p.y)} r={3} fill={s.color} stroke="#fff" strokeWidth={1} />
            ))}
        </g>
      ))}

      {highlightX !== undefined && highlightX !== null && (
        <line
          x1={xScale(highlightX)}
          x2={xScale(highlightX)}
          y1={m.t}
          y2={height - m.b}
          stroke={COLORS.ink}
          strokeWidth={1}
          opacity={0.45}
        />
      )}

      {hover && (
        <g pointerEvents="none">
          <line x1={hover.sx} x2={hover.sx} y1={m.t} y2={height - m.b} stroke={COLORS.grayLight} />
          {hover.pts.map(({ s, p }, i) => (
            <g key={s.id}>
              <circle cx={xScale(p.x)} cy={yScale(p.y)} r={4.5} fill={s.color} stroke="#fff" strokeWidth={1.5} />
              <text
                className="tick-label"
                x={Math.min(xScale(p.x) + 8, width - 60)}
                y={yScale(p.y) - 8 - i * 0}
                fill={s.color}
                style={{ fontWeight: 600 }}
              >
                {(yFmt ?? ((v: number) => v.toFixed(2)))(p.y)}
              </text>
            </g>
          ))}
        </g>
      )}

      <rect
        x={m.l}
        y={m.t}
        width={width - m.l - m.r}
        height={height - m.t - m.b}
        fill="transparent"
        onMouseMove={handleMove}
        onMouseLeave={() => {
          setHover(null)
          onHoverX?.(null)
        }}
      />
    </ChartSvg>
  )
}
