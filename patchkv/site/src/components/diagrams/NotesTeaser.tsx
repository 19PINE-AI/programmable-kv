import { useEffect, useState } from 'react'
import { useInView } from '../../lib/useInView'
import { COLORS } from '../charts/core'

/**
 * The hero animation: at prefill (phase 0) the model writes the
 * field-conditioned conclusion onto downstream aggregator tokens (orange);
 * at decode (phase 1) the decision reads those notes, not the field (blue).
 * Schematic — mirrors fig. 1 of the paper.
 */

const CELLS = [
  { id: 'sys', label: 'system', w: 64 },
  { id: 'field', label: 'status: pending', w: 110, field: true },
  { id: 'rule', label: 'policy rule', w: 86 },
  { id: 'f1', label: '…', w: 36 },
  { id: 'note1', label: '"…nothing else."', w: 104, note: true },
  { id: 'f2', label: '…', w: 36 },
  { id: 'note2', label: 'TASK ¶', w: 64, note: true },
  { id: 'dec', label: 'decision →', w: 86, dec: true },
]

export function NotesTeaser() {
  const [ref, inView] = useInView<HTMLDivElement>(0.4)
  const [phase, setPhase] = useState(0) // 0 prefill-write, 1 decode-read

  useEffect(() => {
    if (!inView) return
    const t = setInterval(() => setPhase((p) => (p + 1) % 2), 3000)
    return () => clearInterval(t)
  }, [inView])

  const W = 760
  const H = 240
  const y = 138
  let x = 32
  const pos: Record<string, { x: number; w: number }> = {}
  for (const c of CELLS) {
    pos[c.id] = { x, w: c.w }
    x += c.w + 10
  }
  const cx = (id: string) => pos[id].x + pos[id].w / 2

  function arc(x1: number, x2: number, up: boolean, k = 0.5) {
    const my = up ? y - 64 - Math.abs(x2 - x1) * 0.06 * k : y + 96
    return `M ${x1} ${up ? y - 16 : y + 30} Q ${(x1 + x2) / 2} ${my} ${x2} ${up ? y - 16 : y + 30}`
  }

  const writeArcs = [
    { d: arc(cx('field'), cx('note1'), true), from: 'field' },
    { d: arc(cx('rule'), cx('note1'), true, 0.8), from: 'rule' },
    { d: arc(cx('field'), cx('note2'), true, 1.2), from: 'field' },
  ]
  const readArcs = [
    { d: arc(cx('dec'), cx('note1'), true), share: '56%' },
    { d: arc(cx('dec'), cx('note2'), true, 0.7), share: '' },
  ]

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <defs>
          <marker id="arrow-o" viewBox="0 0 8 8" refX={6} refY={4} markerWidth={6} markerHeight={6} orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={COLORS.orange} />
          </marker>
          <marker id="arrow-b" viewBox="0 0 8 8" refX={6} refY={4} markerWidth={6} markerHeight={6} orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={COLORS.blue} />
          </marker>
        </defs>

        {/* phase label */}
        <g style={{ transition: 'opacity 0.5s' }}>
          <text x={32} y={34} style={{ fontFamily: 'var(--sans)', fontSize: 13, fontWeight: 600 }}
            fill={phase === 0 ? COLORS.orange : COLORS.blue}>
            {phase === 0 ? 'PREFILL — the model writes notes' : 'DECODE — the decision reads the notes, not the field'}
          </text>
          <text x={32} y={52} style={{ fontFamily: 'var(--sans)', fontSize: 11.5 }} fill="var(--ink-faint)">
            {phase === 0
              ? 'the field-conditioned conclusion is memoized onto downstream aggregator tokens'
              : 'the field itself drives <1% of the decision; the downstream notes drive essentially all of it'}
          </text>
        </g>

        {/* arcs */}
        <g style={{ opacity: phase === 0 ? 1 : 0.08, transition: 'opacity 0.7s' }}>
          {writeArcs.map((a, i) => (
            <path key={i} d={a.d} fill="none" stroke={COLORS.orange} strokeWidth={2}
              markerEnd="url(#arrow-o)" strokeDasharray="6 5"
              style={{ animation: inView && phase === 0 ? 'dashflow 1.4s linear infinite' : undefined }} />
          ))}
        </g>
        <g style={{ opacity: phase === 1 ? 1 : 0.08, transition: 'opacity 0.7s' }}>
          {readArcs.map((a, i) => (
            <path key={i} d={a.d} fill="none" stroke={COLORS.blue} strokeWidth={2.4}
              markerEnd="url(#arrow-b)"
              style={{ animation: inView && phase === 1 ? 'dashflow 1.4s linear infinite' : undefined }}
              strokeDasharray="6 5" />
          ))}
          {/* faint, struck-through arc to the field: the decision does NOT read it */}
          <path d={arc(cx('dec'), cx('field'), false)} fill="none" stroke={COLORS.gray} strokeWidth={1.4} strokeDasharray="3 4" />
          <text x={(cx('dec') + cx('field')) / 2} y={y + 92} textAnchor="middle"
            style={{ fontFamily: 'var(--sans)', fontSize: 10.5 }} fill="var(--ink-faint)">
            direct read of the field: &lt;1% of the decision
          </text>
        </g>

        {/* token cells */}
        {CELLS.map((c) => {
          const p = pos[c.id]
          const isNote = !!c.note
          const noteLit = isNote && (phase === 1 || phase === 0)
          return (
            <g key={c.id}>
              <rect x={p.x} y={y - 16} width={p.w} height={44} rx={6}
                fill={c.field ? 'var(--orange-faint)' : isNote && noteLit ? COLORS.orangeSoft : c.dec ? 'var(--blue-faint)' : '#fff'}
                stroke={c.field ? COLORS.orange : isNote ? COLORS.orange : c.dec ? COLORS.blue : 'var(--rule-strong)'}
                strokeWidth={c.field || isNote || c.dec ? 1.6 : 1}
                style={{ transition: 'fill 0.7s' }} />
              <text x={p.x + p.w / 2} y={y + 8} textAnchor="middle"
                style={{ fontFamily: c.field || isNote ? 'var(--mono)' : 'var(--sans)', fontSize: 10.5, fontWeight: c.field || c.dec ? 600 : 400 }}
                fill="var(--ink-soft)">
                {c.label}
              </text>
              {isNote && (
                <text x={p.x + p.w / 2} y={y - 23} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 9.5, fontWeight: 700 }}
                  fill={COLORS.orange} opacity={phase === 0 ? 1 : 0.75}>
                  ✎ note: “deny”
                </text>
              )}
            </g>
          )
        })}

        {/* substrate label */}
        <text x={32} y={y + 58} style={{ fontFamily: 'var(--sans)', fontSize: 10.5, letterSpacing: '0.08em' }} fill="var(--ink-faint)">
          KV CACHE (ONE ENTRY PER TOKEN, PER LAYER)
        </text>
      </svg>
      <style>{`@keyframes dashflow { to { stroke-dashoffset: -22; } }`}</style>
    </div>
  )
}
