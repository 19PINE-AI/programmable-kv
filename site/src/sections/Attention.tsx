import { Section, P, Aside, PaperConst } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { ChartSvg, COLORS } from '../components/charts/core'
import { fmtPct } from '../lib/format'
import constants from '../data/constants.json'

const META = { id: 'attention', num: '3', title: 'How attention reads the notes' }

/**
 * One panel of the attention-flow comparison: where the decision token's
 * attention mass goes, with (`masked`) or without the knockout of its edges to
 * the stale downstream notes. Static — the two panels are shown stacked.
 */
function FlowPanel({ masked, heading }: { masked: boolean; heading: string }) {
  const shares = constants.attention_shares
  const ko = constants.attention_knockout

  const W = 720
  const y = 96
  const cells = [
    { id: 'sink', label: '⟨bos⟩ sink', x: 24, w: 76 },
    { id: 'field', label: 'FIELD (fresh)', x: 116, w: 104, field: true },
    { id: 'rule', label: 'rule', x: 232, w: 60 },
    { id: 'notes', label: 'stale downstream notes ✎ ✎ ✎', x: 304, w: 224, note: true },
    { id: 'dec', label: 'decision', x: 568, w: 92, dec: true },
  ]
  const pos = Object.fromEntries(cells.map((c) => [c.id, c]))
  const cx = (id: string) => pos[id].x + pos[id].w / 2

  function arc(x1: number, x2: number, lift = 1) {
    return `M ${x1} ${y - 14} Q ${(x1 + x2) / 2} ${y - 56 - 26 * lift} ${x2} ${y - 14}`
  }

  const flows = masked
    ? [
        { to: 'field', share: 0.42, w: 7 },
        { to: 'sink', share: 0.5, w: 8 },
      ]
    : [
        { to: 'notes', share: shares.downstream, w: 9 },
        { to: 'sink', share: shares.sink, w: 6.5 },
        { to: 'field', share: shares.field, w: 1.2 },
      ]

  const pSafe = masked ? ko.non_reasoning.masked_P_safe : ko.non_reasoning.baseline_P_safe

  return (
    <ChartSvg width={W} height={210}>
      <text x={24} y={18} style={{ fontFamily: 'var(--sans)', fontSize: 12.5, fontWeight: 700 }} fill={masked ? COLORS.red : COLORS.blue}>
        {heading}
      </text>

      {flows.map((f) => (
        <g key={f.to}>
          <path d={arc(cx('dec'), cx(f.to), f.to === 'sink' ? 1.5 : f.to === 'field' ? 1.15 : 0.8)} fill="none"
            stroke={f.to === 'notes' ? COLORS.orange : f.to === 'field' ? COLORS.green : COLORS.gray}
            strokeWidth={f.w} opacity={0.75} strokeLinecap="round" />
          <text x={(cx('dec') + cx(f.to)) / 2}
            y={y - 62 - 26 * (f.to === 'sink' ? 1.5 : f.to === 'field' ? 1.15 : 0.8)}
            textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 11.5, fontWeight: 700 }}
            fill={f.to === 'notes' ? COLORS.orange : f.to === 'field' ? COLORS.green : 'var(--ink-faint)'}>
            {f.to === 'notes' ? `notes ${fmtPct(f.share, 0)}` : f.to === 'field' ? (masked ? 'field re-derived' : `field ${fmtPct(f.share, 1)}`) : `sink ${fmtPct(f.share, 0)}`}
          </text>
        </g>
      ))}
      {masked && (
        <g>
          <path d={arc(cx('dec'), cx('notes'), 0.8)} fill="none" stroke={COLORS.red} strokeWidth={1.6} strokeDasharray="4 4" />
          <text x={(cx('dec') + cx('notes')) / 2} y={y - 62 - 20} textAnchor="middle"
            style={{ fontFamily: 'var(--sans)', fontSize: 12, fontWeight: 700 }} fill={COLORS.red}>
            ✕ masked
          </text>
        </g>
      )}

      {cells.map((c: any) => (
        <g key={c.id}>
          <rect x={c.x} y={y - 14} width={c.w} height={44} rx={6}
            fill={c.field ? '#e7f3e9' : c.note ? (masked ? '#f0eee6' : 'var(--orange-faint)') : c.dec ? 'var(--blue-faint)' : '#fff'}
            stroke={c.field ? COLORS.green : c.note ? COLORS.orange : c.dec ? COLORS.blue : 'var(--rule-strong)'}
            strokeWidth={1.5} />
          <text x={c.x + c.w / 2} y={y + 12} textAnchor="middle"
            style={{ fontFamily: 'var(--sans)', fontSize: 10.5, fontWeight: c.field || c.dec ? 600 : 400 }} fill="var(--ink-soft)">
            {c.label}
          </text>
        </g>
      ))}

      <g transform={`translate(24,${y + 64})`}>
        <text style={{ fontFamily: 'var(--sans)', fontSize: 12 }} fill="var(--ink-soft)">
          field-only edit, no chain-of-thought:&nbsp;&nbsp;P(safe decision) =
        </text>
        <text x={392} style={{ fontFamily: 'var(--sans)', fontSize: 15, fontWeight: 700 }}
          fill={pSafe > 0.5 ? COLORS.green : COLORS.red}>
          {pSafe.toFixed(2)} {pSafe > 0.5 ? '✓ follows the fresh field' : '✗ follows the stale notes'}
        </text>
      </g>
    </ChartSvg>
  )
}

function AttentionFlow() {
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      <FlowPanel masked={false} heading="① attention as-is — the decision reads the stale notes" />
      <div style={{ borderTop: '1px dashed var(--rule-strong)' }} />
      <FlowPanel masked heading="② mask the decision’s edges to the stale notes — it flips" />
    </div>
  )
}

export function Attention() {
  const shares = constants.attention_shares
  const ko = constants.attention_knockout
  return (
    <Section meta={META}>
      <P>
        Patching shows <em>where</em> the conclusion lives; attention shows <em>how it is read</em>.
        Measure where the decision token&rsquo;s attention mass actually goes in the gated-decision
        prompt: about {fmtPct(shares.downstream, 0)} lands on the stale downstream region —
        disproportionately on aggregator and delimiter tokens like end-of-rule punctuation and the{' '}
        <code>TASK</code> header — about {fmtPct(shares.sink, 0)} on the attention sink, and{' '}
        <strong>about {fmtPct(shares.field, 1)} on the field itself</strong>. The freshly edited
        field is barely consulted; the decision flows through the notes.
      </P>

      <Figure
        label="Attention knockout."
        title="The decision reads the notes — cut the edges and it stops"
        sub="Top: attention as-is. Bottom: the decision token's attention to the stale downstream positions is masked."
        caption={
          <>
            With attention intact, a field-only edit is ignored (P(safe) ={' '}
            {ko.non_reasoning.baseline_P_safe.toFixed(2)}) — the decision keeps reading the stale
            notes. Masking exactly those attention edges flips it (P(safe) ={' '}
            {ko.non_reasoning.masked_P_safe.toFixed(2)}): starved of its notes, the model falls
            back to re-deriving from the rule and the fresh field. Arc widths are proportional to
            measured attention mass. <PaperConst src={ko.source} />
          </>
        }
      >
        <AttentionFlow />
      </Figure>

      <P>
        The same intervention explains the reasoning fast-path from §1: under chain-of-thought the
        chain itself is the corrective reader — it re-reads the fresh field and re-derives the
        conclusion (P(safe) {ko.reasoning.baseline_P_safe.toFixed(2)} with field-only editing), and
        masking the <em>chain&rsquo;s</em> attention instead drops safety to{' '}
        {ko.reasoning.masked_P_safe.toFixed(2)}. Whether an in-place edit works is exactly the
        question of <em>which reader gets to the answer first</em>: the memoized note or a live
        re-derivation.
      </P>

      <Aside>
        <b>Connection to interpretability.</b> Aggregation onto delimiter tokens is the same
        structural motif reported in Anthropic&rsquo;s circuit work on forward planning (plans
        stored on line-break tokens). Here the stored object is a backward-looking,
        field-conditioned <b>conclusion</b> — and because it lives in the KV cache, an inference
        system can read <em>and write</em> it directly.
      </Aside>
    </Section>
  )
}
