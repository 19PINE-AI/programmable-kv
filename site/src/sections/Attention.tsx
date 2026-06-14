import { Section, P, Aside, PaperConst } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { ChartSvg, COLORS } from '../components/charts/core'
import { fmtPct } from '../lib/format'
import constants from '../data/constants.json'

const META = { id: 'attention', num: '8', title: 'How the model reads its own notes' }

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
    { id: 'sink', label: 'start position', x: 24, w: 76 },
    { id: 'field', label: 'the fresh fact', x: 116, w: 104, field: true },
    { id: 'rule', label: 'rule', x: 232, w: 60 },
    { id: 'notes', label: 'old notes ✎ ✎ ✎', x: 304, w: 224, note: true },
    { id: 'dec', label: 'the answer', x: 568, w: 92, dec: true },
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
            {f.to === 'notes' ? `notes ${fmtPct(f.share, 0)}` : f.to === 'field' ? (masked ? 'reworked from the fact' : `fact ${fmtPct(f.share, 1)}`) : `start ${fmtPct(f.share, 0)}`}
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
          changed the fact only, no step-by-step thinking:&nbsp;&nbsp;chance of a safe decision =
        </text>
        <text x={392} style={{ fontFamily: 'var(--sans)', fontSize: 15, fontWeight: 700 }}
          fill={pSafe > 0.5 ? COLORS.green : COLORS.red}>
          {pSafe.toFixed(2)} {pSafe > 0.5 ? '✓ follows the fresh fact' : '✗ follows the old notes'}
        </text>
      </g>
    </ChartSvg>
  )
}

function AttentionFlow() {
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      <FlowPanel masked={false} heading="① normal: the answer is pulled from the stale notes" />
      <div style={{ borderTop: '1px dashed var(--rule-strong)' }} />
      <FlowPanel masked heading="② block the look-back to those notes — the answer flips" />
    </div>
  )
}

export function Attention() {
  const shares = constants.attention_shares
  const ko = constants.attention_knockout
  return (
    <Section meta={META}>
      <P>
        Earlier we found <em>where</em> the model&rsquo;s conclusion is stored. Here we watch{' '}
        <em>how it reads that conclusion back</em>. As a model writes its answer, it looks back at
        earlier spots in the text and pulls information from them &mdash; researchers call this
        &ldquo;paying attention.&rdquo; So we measured where the model actually looks when it makes
        the decision. About {fmtPct(shares.downstream, 0)} of its look-back lands on the spots where
        it took notes earlier &mdash; especially on small marker positions like the punctuation that
        ends a rule and the <code>TASK</code> header. About {fmtPct(shares.sink, 0)} lands on a
        catch-all starting position, and{' '}
        <strong>only about {fmtPct(shares.field, 1)} on the fact we actually changed</strong>. The
        model barely glances at the fresh fact; its answer comes from the old notes.
      </P>

      <Figure
        label="Blocking the look-back."
        title="The answer comes from the notes — block the look-back and it stops"
        sub="Top: how the model normally looks back. Bottom: we block its look-back to the old note positions."
        caption={
          <>
            Normally, when we change only the fact, the model ignores the change (the chance of a
            safe decision is {ko.non_reasoning.baseline_P_safe.toFixed(2)}) &mdash; it keeps reading
            its old notes. Block just that look-back and the answer flips (the chance of a safe
            decision rises to {ko.non_reasoning.masked_P_safe.toFixed(2)}): with no notes to lean
            on, the model works the answer out again from the rule and the fresh fact. The thicker
            the arc, the more the model looks there. <PaperConst src={ko.source} />
          </>
        }
      >
        <AttentionFlow />
      </Figure>

      <P>
        This also explains the shortcut we saw in §6 when the model thinks step by step. In that
        case, the step-by-step thinking does the corrective work: it re-reads the fresh fact and
        works out the answer again (the chance of a safe decision is{' '}
        {ko.reasoning.baseline_P_safe.toFixed(2)} when we change only the fact). And if we instead
        block the <em>thinking</em> from looking back, safety drops to{' '}
        {ko.reasoning.masked_P_safe.toFixed(2)}. So whether changing a fact in place actually works
        comes down to one race: <em>which answer arrives first</em> &mdash; the one copied from the
        old notes, or the one worked out fresh.
      </P>

      <Aside>
        <b>A link to other research.</b> The model storing information on small marker positions
        is the same pattern Anthropic found when studying how models plan ahead (they stash plans
        on line-break positions). The difference here: what gets stored is a finished{' '}
        <b>conclusion</b> that depends on the fact. And because it sits in the model&rsquo;s
        notebook of notes &mdash; the running scratchpad it keeps while reading &mdash; a system can
        read <em>and rewrite</em> it directly.
      </Aside>
    </Section>
  )
}
