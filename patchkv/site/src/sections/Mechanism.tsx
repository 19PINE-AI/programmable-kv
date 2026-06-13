import { useMemo, useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH } from '../components/charts/BarCI'
import { ChartSvg, COLORS, Legend } from '../components/charts/core'
import { fmt, fmtPct } from '../lib/format'
import mechanism from '../data/mechanism.json'

const META = { id: 'mechanism', num: '2', title: 'The discovery: memoized inference, in four causal probes' }

type CacheState = 'stale' | 'field_only' | 'full_downstream' | 'oracle'

const STATES: { key: CacheState; label: string }[] = [
  { key: 'stale', label: 'stale' },
  { key: 'field_only', label: 'field-only' },
  { key: 'full_downstream', label: 'full-downstream' },
  { key: 'oracle', label: 'oracle' },
]

function CachePatchLab() {
  const models = mechanism.models as any[]
  const [tag, setTag] = useState('qwen3_8b')
  const [state, setState] = useState<CacheState>('field_only')
  const m = models.find((x) => x.tag === tag)!

  const recovery: number =
    state === 'stale' ? 0 : state === 'oracle' ? 1 :
    state === 'field_only' ? m.field_only.mean : m.full_downstream.mean
  const ci: [number, number] | null =
    state === 'field_only' ? m.field_only.ci : state === 'full_downstream' ? m.full_downstream.ci : null

  const fieldFresh = state !== 'stale'
  const downFresh = state === 'full_downstream' || state === 'oracle'
  const prefixFresh = state === 'oracle'

  const W = 700
  const cellY = 60
  const cellH = 52

  function region(x: number, w: number, label: string, fresh: boolean, kind: 'prefix' | 'field' | 'down') {
    const fill = fresh ? '#e7f3e9' : kind === 'field' ? 'var(--orange-faint)' : '#f0eee6'
    const stroke = fresh ? COLORS.green : kind === 'field' ? COLORS.orange : 'var(--rule-strong)'
    return (
      <g>
        <rect x={x} y={cellY} width={w} height={cellH} rx={6} fill={fill} stroke={stroke} strokeWidth={1.6} style={{ transition: 'fill .4s, stroke .4s' }} />
        <text x={x + w / 2} y={cellY + 22} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 11.5, fontWeight: 600 }} fill="var(--ink)">
          {label}
        </text>
        <text x={x + w / 2} y={cellY + 38} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 10 }} fill={fresh ? COLORS.green : 'var(--ink-faint)'}>
          {fresh ? '↻ recomputed with NEW value' : kind === 'prefix' ? 'reused (deviation 0.0)' : 'stale (attended to OLD value)'}
        </text>
        {kind === 'down' && !fresh && (
          <g fill={COLORS.orange}>
            {[0.18, 0.46, 0.8].map((f) => (
              <text key={f} x={x + w * f} y={cellY - 6} textAnchor="middle" style={{ fontSize: 11 }}>✎</text>
            ))}
          </g>
        )}
      </g>
    )
  }

  const followsNew = recovery > 0.5
  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={models.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
        <ControlGroup label="cache state">
          <Seg options={STATES.map((s) => s.key) as any} value={state} onChange={(v) => setState(v as CacheState)}
            labels={Object.fromEntries(STATES.map((s) => [s.key, s.label])) as any} />
        </ControlGroup>
      </Controls>

      <ChartSvg width={W} height={172}>
        {region(10, 200, 'prefix (before the field)', prefixFresh, 'prefix')}
        {region(218, 96, 'FIELD', fieldFresh, 'field')}
        {region(322, 240, 'downstream tokens', downFresh, 'down')}
        {/* decision readout */}
        <g>
          <rect x={584} y={cellY - 6} width={106} height={cellH + 12} rx={8} fill="var(--blue-faint)" stroke={COLORS.blue} strokeWidth={1.6} />
          <text x={637} y={cellY + 14} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 11, fontWeight: 600 }} fill={COLORS.blue}>
            decision
          </text>
          <text x={637} y={cellY + 33} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700 }}
            fill={followsNew ? COLORS.green : COLORS.red}>
            {followsNew ? 'NEW value' : 'OLD value'}
          </text>
          <text x={637} y={cellY + 48} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 9.5 }} fill="var(--ink-faint)">
            {followsNew ? 'follows the edit' : 'as if nothing changed'}
          </text>
        </g>
        <path d={`M 568 ${cellY + cellH / 2} L 580 ${cellY + cellH / 2}`} stroke={COLORS.blue} strokeWidth={2} markerEnd="none" />

        {/* recovery gauge */}
        <g transform="translate(10,138)">
          <text x={0} y={4} style={{ fontFamily: 'var(--sans)', fontSize: 11, fontWeight: 600 }} fill="var(--ink-soft)">
            decision recovery
          </text>
          <rect x={130} y={-7} width={420} height={14} rx={7} fill="#eceadf" />
          <rect x={130 + Math.min(Math.max(recovery, 0), 1) * 0 } y={-7} width={Math.min(Math.max(recovery, 0), 1) * 420} height={14} rx={7}
            fill={followsNew ? COLORS.green : COLORS.orange} style={{ transition: 'width .5s' }} />
          {ci && (
            <g stroke="var(--ink)" strokeWidth={1.2} opacity={0.55}>
              <line x1={130 + Math.max(0, ci[0]) * 420} x2={130 + Math.min(1, ci[1]) * 420} y1={0} y2={0} />
            </g>
          )}
          <text x={562} y={4} style={{ fontFamily: 'var(--sans)', fontSize: 13, fontWeight: 700 }} fill="var(--ink)">
            {fmt(recovery, 3)}
          </text>
        </g>
      </ChartSvg>

      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', lineHeight: 1.6 }}>
        {state === 'stale' && <>Reuse everything: the decision behaves like the old world — recovery 0 by definition.</>}
        {state === 'field_only' && (
          <>
            Refresh the field&rsquo;s own KV and reuse the rest: recovery{' '}
            <b>{fmt(m.field_only.mean, 3)}</b>{m.field_only.ci && <> (CI {fmt(m.field_only.ci[0], 3)}…{fmt(m.field_only.ci[1], 3)})</>} on {m.label} —
            essentially none of the oracle&rsquo;s decision flip. The fresh field is ignored.
          </>
        )}
        {state === 'full_downstream' && (
          <>
            Keep the field <em>stale</em> but recompute everything after it: recovery{' '}
            <b>{fmt(m.full_downstream.mean, 3)}</b>. The information the decision needs lives downstream.
          </>
        )}
        {state === 'oracle' && <>Clean prefill of the new value — recovery 1 by definition.</>}
      </div>
    </div>
  )
}

function FieldVsDownstreamStrip() {
  const models = (mechanism.models as any[]).filter((m) => !m.tag.includes('int8'))
  return (
    <BarsH
      items={models.map((m) => ({
        label: m.label,
        value: m.field_only.mean,
        lo: m.field_only.ci?.[0],
        hi: m.field_only.ci?.[1],
        color: COLORS.orange,
        marker: m.full_downstream.mean,
      }))}
      domain={[-0.1, 1.1]}
      xLabel="decision recovery"
      refX={[{ x: 0, label: 'stale' }, { x: 1, label: 'oracle' }]}
      markerLabel="full-downstream recovery"
      labelWidth={170}
    />
  )
}

function SuffixConcentration() {
  const models = (mechanism.models as any[]).filter((m) => m.cum_suffix.length > 0)
  const [tag, setTag] = useState('qwen3_8b')
  const [frac, setFrac] = useState(0.2)
  const m = models.find((x) => x.tag === tag)!

  const nearest = (m.cum_suffix as any[]).reduce((a, b) => (Math.abs(b.frac - frac) < Math.abs(a.frac - frac) ? b : a))
  const nearestPre = (m.cum_prefix as any[]).reduce((a, b) => (Math.abs(b.frac - frac) < Math.abs(a.frac - frac) ? b : a))

  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={models.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
        <ControlGroup label="patched fraction">
          <input type="range" className="slider" min={0.02} max={1} step={0.01} value={frac}
            onChange={(e) => setFrac(parseFloat(e.target.value))} style={{ width: 180 }} />
          <span className="readout" style={{ marginLeft: 8 }}>{fmtPct(frac, 0)}</span>
        </ControlGroup>
      </Controls>
      <LineChart
        series={[
          {
            id: 'suffix', label: 'patch downstream tokens (ranked by effect)', color: COLORS.orange,
            points: (m.cum_suffix as any[]).map((p) => ({ x: p.frac, y: p.mean, lo: p.ci?.[0], hi: p.ci?.[1] })), band: true,
          },
          {
            id: 'prefix', label: 'patch prefix tokens (control)', color: COLORS.gray, dash: true,
            points: (m.cum_prefix as any[]).map((p) => ({ x: p.frac, y: p.mean, lo: p.ci?.[0], hi: p.ci?.[1] })),
          },
        ]}
        xLabel="fraction of tokens patched with fresh KV"
        yLabel="decision recovery"
        yDomain={[-0.08, 1.08]}
        xFmt={(v) => fmtPct(v, 0)}
        refLinesY={[{ y: 1, label: 'oracle' }, { y: 0, label: 'stale' }]}
        highlightX={frac}
        height={300}
      />
      <Legend items={[
        { label: 'downstream (suffix) tokens', color: COLORS.orange },
        { label: 'prefix tokens — control', color: COLORS.gray, dash: true },
      ]} />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        At {fmtPct(frac, 0)} patched: downstream recovers <b>{fmt(nearest.mean, 2)}</b> of the decision; the same
        fraction of prefix tokens recovers <b>{fmt(nearestPre.mean, 2)}</b>. The effect lives after the field,
        spread over many tokens.
      </div>
    </div>
  )
}

function DoseResponse() {
  const dose = mechanism.dose as any[]
  return (
    <BarsH
      items={dose.map((d) => ({
        label: d.label, value: d.mean, lo: d.ci?.[0], hi: d.ci?.[1], color: COLORS.blue,
      }))}
      domain={[-0.05, 1.05]}
      xLabel="field-only recovery (Qwen3-8B)"
      refX={[{ x: 1, label: 'oracle' }]}
      labelWidth={190}
    />
  )
}

function WordingAblation() {
  const w = mechanism.wording as any[]
  const order = ['none', 'value_only', 'update_tag', 'override_full', 'conclusion']
  const items = order.map((k) => w.find((x) => x.key === k)!).filter(Boolean)
  return (
    <BarsH
      items={items.map((x) => ({
        label: x.label, value: x.P_safe, lo: x.ci?.[0], hi: x.ci?.[1],
        color: x.key === 'none' ? COLORS.gray : x.key === 'conclusion' ? COLORS.red : COLORS.green,
      }))}
      domain={[0, 1.05]}
      xLabel={`P(safe decision) — corrected value in context + appended wording (Qwen3-8B, n=${mechanism.wording_n})`}
      labelWidth={215}
    />
  )
}

export function Mechanism() {
  const q8 = (mechanism.models as any[]).find((m) => m.tag === 'qwen3_8b')!
  return (
    <Section meta={META}>
      <P>
        The method is <strong>causal patching</strong> on the KV cache itself: prefill the context
        twice (old and new field value), then build hybrid caches that mix entries from the two
        worlds and read which decision comes out. We report <em>decision recovery</em> — the
        fraction of the oracle&rsquo;s decision flip a cache state reproduces (0 = behaves stale,
        1 = behaves like a clean prefill). Try the four cache states the paper compares:
      </P>

      <Figure
        label="Probe 1 — locality."
        title="Patch the cache, read the decision"
        sub="Interactive: pick a model and a cache state; the recovery numbers are the released causal-patching records"
        caption={
          <>
            Refreshing the field&rsquo;s own KV recovers essentially none of the decision
            (−0.028…0.14 across eleven models); recomputing the downstream while keeping the field
            <em> stale</em> recovers it fully. The field is read indirectly — its causal share of
            the decision is under 1%.
          </>
        }
      >
        <CachePatchLab />
      </Figure>

      <Figure
        narrow
        label="All models."
        caption={
          <>
            Field-only recovery (orange bars, with bootstrap CIs) vs. full-downstream recovery
            (orange diamonds at ≈1.0) for every model in the study — four architecture families,
            0.6B to 32B. The dissociation is universal.
          </>
        }
      >
        <FieldVsDownstreamStrip />
      </Figure>

      <H3>Probe 2 — the effect is suffix-concentrated</H3>
      <P>
        Where downstream does the old conclusion live? Patch fresh KV into an increasing fraction
        of tokens, ranked by causal effect, and watch recovery accrue. Drag the scrubber:
      </P>
      <Figure
        label="Probe 2 — suffix concentration."
        caption={
          <>
            Recovery accrues only as many post-field tokens are patched and saturates near 100% of
            the suffix; patching the same fraction of <em>prefix</em> tokens (control) does
            nothing. On Qwen3-8B, patching the top 2% of downstream tokens already recovers{' '}
            {fmt((q8.cum_suffix as any[])[0].mean, 2)}.
          </>
        }
      >
        <SuffixConcentration />
      </Figure>

      <H3>Probe 3 — the conclusion is written down, not just decodable</H3>
      <P>
        A linear probe finds the field-conditioned conclusion decodable from downstream
        tokens&rsquo; residual streams already at prefill time — the model has computed and
        recorded the answer before any decoding starts (per-layer curves in §4, where we also pin
        down <em>when</em> the note is written). And the dose–response below shows the memoization
        follows the field&rsquo;s position: the later the field appears, the fewer tokens sit
        after it to memoize a conclusion, and the more an in-place edit recovers.
      </P>
      <Figure
        narrow
        label="Probe 4 — dose–response by field position."
        caption={
          <>
            Field-only recovery as the field moves later in the context (Qwen3-8B). Hoisted to the
            end — the de-facto industry workaround — nothing sits after the field, no conclusion
            is memoized, and the in-place edit works. The workaround is a special case of the
            mechanism.
          </>
        }
      >
        <DoseResponse />
      </Figure>

      <H3>What the note contains</H3>
      <P>
        If the note were a verbatim copy of the field, wording appended after it would be inert.
        The ablation isolates pure wording: every variant runs on a <em>clean</em> context that
        already contains the corrected value, plus one appended line. Restating the value or
        tagging an update changes nothing — but aggressively telling the model that its{' '}
        <em>&ldquo;earlier conclusion is void — re-evaluate from scratch&rdquo;</em> actively
        hurts:
      </P>
      <Figure
        narrow
        label="Wording ablation."
        caption={
          <>
            P(safe) with the corrected value in context plus an appended line (Qwen3-8B). Nothing
            appended, bare value, and tagged update sit at 1.0; explicit override is statistically
            indistinguishable; the confrontational re-evaluate phrasing drops to{' '}
            {fmt((mechanism.wording as any[]).find((x) => x.key === 'conclusion')!.P_safe, 2)}.
            The note behaves like a <b>committed conclusion</b>: a late, salient correction can
            overwrite it, but instructions that attack it destabilize the decision.
          </>
        }
      >
        <WordingAblation />
      </Figure>

      <Aside>
        <b>Naming it.</b> At prefill the model computes the field-conditioned conclusion and
        writes it onto downstream aggregator tokens; at decode the decision reads those notes, not
        the field. The paper names this <b>attention-mediated memoized inference</b>. §4 and §5
        stress-test the account; §6–§7 turn it into capabilities.
      </Aside>
    </Section>
  )
}
