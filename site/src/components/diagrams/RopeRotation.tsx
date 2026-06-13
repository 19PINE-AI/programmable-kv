import { useEffect, useState } from 'react'
import { useInView } from '../../lib/useInView'
import { COLORS } from '../charts/core'

/**
 * Schematic of position-portable transplant: a skill's KV is precompiled at
 * source positions 0..L-1; its post-RoPE keys are re-rotated to the target
 * offset (values are position-free) and spliced into the live cache.
 */
export function RopeRotation() {
  const [ref, inView] = useInView<HTMLDivElement>(0.4)
  const [t, setT] = useState(0) // 0..1 animation progress

  useEffect(() => {
    if (!inView) return
    let raf = 0
    let start: number | null = null
    const loop = (ts: number) => {
      if (start === null) start = ts
      const cycle = ((ts - start) % 3600) / 3600
      // ease in-out, hold at ends
      const p = cycle < 0.15 ? 0 : cycle > 0.65 ? 1 : (cycle - 0.15) / 0.5
      setT(p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [inView])

  const W = 720
  const H = 250
  const srcY = 54
  const tgtY = 168
  const cell = 34
  const skillLen = 6
  const srcX = 70
  const tgtX = 330
  const dx = tgtX - srcX
  const dy = tgtY - srcY

  // key "clock" angles: angle proportional to position; re-rotation adds Δpos
  const baseAngle = (i: number) => i * 28
  const delta = 8 * 28 // moving from position 0.. to position 8..

  return (
    <div ref={ref}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {/* source: precompiled in isolation */}
        <text x={srcX} y={srcY - 26} style={{ fontFamily: 'var(--sans)', fontSize: 11.5, fontWeight: 600 }} fill={COLORS.orange}>
          skill precompiled in isolation (positions 0…{skillLen - 1}) — once, offline
        </text>
        {/* target context */}
        <text x={40} y={tgtY - 26} style={{ fontFamily: 'var(--sans)', fontSize: 11.5, fontWeight: 600 }} fill={COLORS.blue}>
          live context (splice at positions 8…{8 + skillLen - 1}) — no recompute
        </text>

        {/* live prefix cells */}
        {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
          <g key={`p${i}`}>
            <rect x={40 + i * (cell + 2)} y={tgtY} width={cell} height={cell} rx={4} fill="#fff" stroke="var(--rule-strong)" />
            <text x={40 + i * (cell + 2) + cell / 2} y={tgtY + cell + 13} textAnchor="middle" className="tick-label" style={{ fontSize: 8.5 }}>{i}</text>
          </g>
        ))}
        {/* suffix cells after the splice */}
        {[0, 1].map((i) => (
          <rect key={`s${i}`} x={40 + (8 + skillLen + i) * (cell + 2) + 8} y={tgtY} width={cell} height={cell} rx={4} fill="#fff" stroke="var(--rule)" strokeDasharray="3 3" />
        ))}

        {/* skill tokens flying from source row to target slot, keys rotating */}
        {Array.from({ length: skillLen }, (_, i) => {
          const x0 = srcX + i * (cell + 2)
          const x1 = 40 + (8 + i) * (cell + 2) + 8
          const xx = x0 + (x1 - x0 + dx * 0) * t - 0 // interpolate
          const cxp = x0 + (x1 - x0) * t
          const cyp = srcY + dy * t
          const ang = baseAngle(i) + delta * t
          const r = cell / 2 - 6
          return (
            <g key={i}>
              {/* ghost at source */}
              <rect x={x0} y={srcY} width={cell} height={cell} rx={4} fill="var(--orange-faint)" stroke={COLORS.orangeSoft} opacity={1 - t * 0.7} />
              <g transform={`translate(${cxp + cell / 2},${cyp + cell / 2})`}>
                <rect x={-cell / 2} y={-cell / 2} width={cell} height={cell} rx={4}
                  fill="var(--orange-faint)" stroke={COLORS.orange} strokeWidth={1.4} />
                {/* the key vector as a clock hand: rotates by Δpos */}
                <line x1={0} y1={0} x2={r * Math.cos((ang * Math.PI) / 180)} y2={r * Math.sin((ang * Math.PI) / 180)}
                  stroke={COLORS.orange} strokeWidth={2.2} strokeLinecap="round" />
                <circle r={2} fill={COLORS.orange} />
              </g>
              <text x={x1 + cell / 2} y={tgtY + cell + 13} textAnchor="middle" className="tick-label" style={{ fontSize: 8.5, fill: COLORS.orange }}>
                {8 + i}
              </text>
            </g>
          )
        })}

        {/* annotation */}
        <g transform={`translate(${srcX + skillLen * (cell + 2) + 18},${srcY + 8})`} opacity={0.95}>
          <text style={{ fontFamily: 'var(--sans)', fontSize: 11 }} fill="var(--ink-soft)">
            keys: re-rotate by Δpos = +8 <tspan style={{ fontStyle: 'italic' }}>(RoPE is a rotation —</tspan>
          </text>
          <text y={15} style={{ fontFamily: 'var(--sans)', fontSize: 11, fontStyle: 'italic' }} fill="var(--ink-soft)">
            moving a token just turns its clock hands)
          </text>
          <text y={34} style={{ fontFamily: 'var(--sans)', fontSize: 11 }} fill="var(--ink-soft)">
            values: copied as-is (position-free)
          </text>
        </g>

        <text x={40} y={H - 6} style={{ fontFamily: 'var(--sans)', fontSize: 10.5 }} fill="var(--ink-faint)">
          cost: one O(L) re-rotation pass over the skill&rsquo;s keys + prefill of the short suffix — vs. O(L²) attention to re-prefill the skill
        </text>
      </svg>
    </div>
  )
}
