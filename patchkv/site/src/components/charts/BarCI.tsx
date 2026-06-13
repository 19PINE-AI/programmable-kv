import { scaleLinear, scaleBand } from 'd3-scale'
import { AxisBottom, AxisLeft, ChartSvg, COLORS } from './core'

export interface BarItem {
  label: string
  value: number | null
  lo?: number
  hi?: number
  color?: string
  note?: string // small annotation rendered after the value
  marker?: number | null // optional secondary value rendered as a diamond marker
}

/** Horizontal bars with CI whiskers — the workhorse comparison chart. */
export function BarsH({
  items,
  width = 620,
  domain,
  xLabel,
  xFmt = (v) => v.toFixed(2),
  valueFmt,
  refX = [],
  barColor = COLORS.blue,
  markerLabel,
  labelWidth = 150,
}: {
  items: BarItem[]
  width?: number
  domain?: [number, number]
  xLabel?: string
  xFmt?: (v: number) => string
  valueFmt?: (v: number) => string
  refX?: { x: number; label?: string; color?: string }[]
  barColor?: string
  markerLabel?: string
  labelWidth?: number
}) {
  const rowH = 26
  const m = { l: labelWidth, r: 56, t: 8, b: 40 }
  const height = m.t + m.b + items.length * rowH
  const vals = items.flatMap((it) => [it.value ?? 0, it.lo ?? 0, it.hi ?? 0, it.marker ?? 0])
  const xd = domain ?? [Math.min(0, ...vals), Math.max(...vals) * 1.08]
  const x = scaleLinear().domain(xd).range([m.l, width - m.r])
  const vf = valueFmt ?? xFmt

  return (
    <ChartSvg width={width} height={height}>
      <AxisBottom scale={x} y={height - m.b} ticks={x.ticks(5)} fmt={xFmt} label={xLabel} grid gridY1={m.t} gridY2={height - m.b} />
      {refX.map((r, i) => (
        <g key={i}>
          <line x1={x(r.x)} x2={x(r.x)} y1={m.t} y2={height - m.b} stroke={r.color ?? COLORS.gray} strokeDasharray="3 4" />
          {r.label && (
            <text className="tick-label" x={x(r.x)} y={m.t + 2} dy="-0.3em" textAnchor="middle">
              {r.label}
            </text>
          )}
        </g>
      ))}
      {items.map((it, i) => {
        const yc = m.t + i * rowH + rowH / 2
        const clamp = (px: number) => Math.max(m.l, Math.min(width - m.r, px))
        const v = it.value === null ? null : Math.max(xd[0], Math.min(xd[1], it.value))
        const zero = clamp(x(Math.max(0, xd[0])))
        return (
          <g key={it.label}>
            <text className="tick-label" x={m.l - 8} y={yc} dy="0.32em" textAnchor="end" style={{ fontSize: 11.5 }}>
              {it.label}
            </text>
            {v !== null && (
              <>
                <rect
                  x={Math.min(zero, x(v))}
                  y={yc - 8}
                  width={Math.abs(x(v) - zero)}
                  height={16}
                  fill={it.color ?? barColor}
                  rx={2}
                />
                {it.lo !== undefined && it.hi !== undefined && (
                  <g stroke={COLORS.ink} strokeWidth={1.1} opacity={0.65}>
                    <line x1={clamp(x(it.lo))} x2={clamp(x(it.hi))} y1={yc} y2={yc} />
                    <line x1={clamp(x(it.lo))} x2={clamp(x(it.lo))} y1={yc - 4} y2={yc + 4} />
                    <line x1={clamp(x(it.hi))} x2={clamp(x(it.hi))} y1={yc - 4} y2={yc + 4} />
                  </g>
                )}
                <text
                  className="tick-label"
                  x={clamp(Math.max(x(it.hi ?? v), x(v))) + 6}
                  y={yc}
                  dy="0.32em"
                  style={{ fontWeight: 600, fill: 'var(--ink)' }}
                >
                  {vf(it.value!)}
                  {it.note ? ` ${it.note}` : ''}
                </text>
              </>
            )}
            {v === null && (
              <text className="tick-label" x={zero + 6} y={yc} dy="0.32em" style={{ fontStyle: 'italic' }}>
                n/a
              </text>
            )}
            {it.marker !== undefined && it.marker !== null && (
              <path
                d={`M ${x(it.marker)} ${yc - 6} l 6 6 l -6 6 l -6 -6 z`}
                fill={COLORS.orange}
                stroke="#fff"
                strokeWidth={1}
              >
                <title>{markerLabel ? `${markerLabel}: ${it.marker}` : String(it.marker)}</title>
              </path>
            )}
          </g>
        )
      })}
    </ChartSvg>
  )
}

/** Vertical grouped bars with CI whiskers (few categories, 1-3 series). */
export function BarsV({
  groups,
  seriesLabels,
  colors,
  width = 620,
  height = 260,
  yDomain,
  yLabel,
  yFmt = (v) => v.toFixed(2),
}: {
  groups: { label: string; values: ({ v: number; lo?: number; hi?: number } | null)[] }[]
  seriesLabels: string[]
  colors: string[]
  width?: number
  height?: number
  yDomain?: [number, number]
  yLabel?: string
  yFmt?: (v: number) => string
}) {
  const m = { l: 56, r: 12, t: 12, b: 56 }
  const all = groups.flatMap((g) => g.values.filter(Boolean).flatMap((x) => [x!.v, x!.lo ?? x!.v, x!.hi ?? x!.v]))
  const yd = yDomain ?? [Math.min(0, ...all), Math.max(...all) * 1.08]
  const y = scaleLinear().domain(yd).range([height - m.b, m.t])
  const x0 = scaleBand<string>()
    .domain(groups.map((g) => g.label))
    .range([m.l, width - m.r])
    .paddingInner(0.32)
    .paddingOuter(0.12)
  const x1 = scaleBand<number>()
    .domain(seriesLabels.map((_, i) => i))
    .range([0, x0.bandwidth()])
    .padding(0.12)

  return (
    <ChartSvg width={width} height={height}>
      <AxisLeft scale={y} x={m.l} ticks={y.ticks(5)} fmt={yFmt} label={yLabel} grid gridX2={width - m.r} />
      <line x1={m.l} x2={width - m.r} y1={y(Math.max(0, yd[0]))} y2={y(Math.max(0, yd[0]))} stroke={COLORS.grayLight} />
      {groups.map((g) => (
        <g key={g.label} transform={`translate(${x0(g.label)},0)`}>
          {g.values.map((val, i) =>
            val === null ? null : (
              <g key={i}>
                <rect
                  x={x1(i)}
                  y={Math.min(y(val.v), y(Math.max(0, yd[0])))}
                  width={x1.bandwidth()}
                  height={Math.abs(y(val.v) - y(Math.max(0, yd[0])))}
                  fill={colors[i]}
                  rx={2}
                />
                {val.lo !== undefined && val.hi !== undefined && (
                  <g stroke={COLORS.ink} strokeWidth={1.1} opacity={0.6}>
                    <line
                      x1={x1(i)! + x1.bandwidth() / 2}
                      x2={x1(i)! + x1.bandwidth() / 2}
                      y1={y(val.lo)}
                      y2={y(val.hi)}
                    />
                  </g>
                )}
                <text
                  className="tick-label"
                  x={x1(i)! + x1.bandwidth() / 2}
                  y={y(Math.max(val.v, val.hi ?? val.v)) - 5}
                  textAnchor="middle"
                  style={{ fontWeight: 600, fill: 'var(--ink)', fontSize: 10 }}
                >
                  {yFmt(val.v)}
                </text>
              </g>
            ),
          )}
          <text
            className="tick-label"
            x={x0.bandwidth() / 2}
            y={height - m.b + 16}
            textAnchor="middle"
            style={{ fontSize: 11 }}
          >
            {g.label}
          </text>
        </g>
      ))}
      <g>
        {seriesLabels.map((s, i) => (
          <g key={s} transform={`translate(${m.l + i * 150},${height - 14})`}>
            <rect width={11} height={11} fill={colors[i]} rx={2} />
            <text className="tick-label" x={16} y={9}>
              {s}
            </text>
          </g>
        ))}
      </g>
    </ChartSvg>
  )
}
