import { useMemo, useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH } from '../components/charts/BarCI'
import { ChartSvg, COLORS, Legend } from '../components/charts/core'
import { fmt, fmtPct } from '../lib/format'
import mechanism from '../data/mechanism.json'

const META = { id: 'mechanism', num: '7', title: 'Models take notes at prefill' }

function CachePatchLab() {
  const models = mechanism.models as any[]
  const [tag, setTag] = useState('qwen3_8b')
  const m = models.find((x) => x.tag === tag)!

  const rows = [
    { label: 'stale', sub: 'reuse everything', prefix: false, field: false, down: false, rec: 0, ci: null as [number, number] | null, byDef: true },
    { label: 'field-only', sub: 'refresh field KV only (~0.2%)', prefix: false, field: true, down: false, rec: m.field_only.mean as number, ci: m.field_only.ci as [number, number] | null },
    { label: 'full-downstream', sub: 'field stale; recompute after it', prefix: false, field: false, down: true, rec: m.full_downstream.mean as number, ci: m.full_downstream.ci as [number, number] | null },
    { label: 'oracle', sub: 'clean prefill of the new value', prefix: true, field: true, down: true, rec: 1, ci: null, byDef: true },
  ]

  const W = 720
  const headH = 30
  const rowH = 62
  const H = headH + rows.length * rowH + 4
  const stripX = 132
  const pW = 138 // prefix cell
  const fW = 62 // field cell
  const dW = 150 // downstream cell
  const barX0 = 556
  const barW = 108

  function cell(x: number, w: number, yc: number, fresh: boolean, kind: 'prefix' | 'field' | 'down', label: string) {
    const fill = fresh ? '#e7f3e9' : kind === 'field' ? 'var(--orange-faint)' : '#f0eee6'
    const stroke = fresh ? COLORS.green : kind === 'field' ? COLORS.orange : 'var(--rule-strong)'
    return (
      <g>
        <rect x={x} y={yc - 17} width={w} height={34} rx={5} fill={fill} stroke={stroke} strokeWidth={1.4} />
        <text x={x + w / 2} y={yc + 1} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 10, fontWeight: 600 }} fill="var(--ink-soft)">
          {label}
        </text>
        {kind === 'down' && !fresh && (
          <g fill={COLORS.orange}>
            {[0.28, 0.72].map((f) => (
              <text key={f} x={x + w * f} y={yc - 21} textAnchor="middle" style={{ fontSize: 10 }}>✎</text>
            ))}
          </g>
        )}
      </g>
    )
  }

  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={models.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
      </Controls>

      <ChartSvg width={W} height={H}>
        {/* column headers */}
        <text x={8} y={16} style={{ fontFamily: 'var(--sans)', fontSize: 10.5, fontWeight: 600 }} fill="var(--ink-faint)">cache state</text>
        <text x={stripX} y={16} style={{ fontFamily: 'var(--sans)', fontSize: 10.5, fontWeight: 600 }} fill="var(--ink-faint)">
          prefix · field · downstream  (green = recomputed)
        </text>
        <text x={barX0} y={16} style={{ fontFamily: 'var(--sans)', fontSize: 10.5, fontWeight: 600 }} fill="var(--ink-faint)">decision · recovery</text>

        {rows.map((r, i) => {
          const yc = headH + i * rowH + rowH / 2 - 4
          const followsNew = r.rec > 0.5
          const fdW = Math.max(0, Math.min(r.rec, 1)) * barW
          const fieldX = stripX + pW + 4
          const downX = fieldX + fW + 4
          return (
            <g key={r.label}>
              {i > 0 && <line x1={8} x2={W - 8} y1={headH + i * rowH} y2={headH + i * rowH} stroke="var(--rule)" />}
              {/* state label */}
              <text x={8} y={yc - 2} style={{ fontFamily: 'var(--sans)', fontSize: 12.5, fontWeight: 700 }} fill="var(--ink)">{r.label}</text>
              <text x={8} y={yc + 13} style={{ fontFamily: 'var(--sans)', fontSize: 9.5 }} fill="var(--ink-faint)">{r.sub}</text>
              {/* cache strip */}
              {cell(stripX, pW, yc, r.prefix, 'prefix', r.prefix ? 'fresh' : 'reused (dev 0.0)')}
              {cell(fieldX, fW, yc, r.field, 'field', 'FIELD')}
              {cell(downX, dW, yc, r.down, 'down', r.down ? 'recomputed' : 'stale notes')}
              {/* decision chip */}
              <text x={downX + dW + 30} y={yc + 4} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 700 }} fill={followsNew ? COLORS.green : COLORS.red}>
                {followsNew ? 'NEW' : 'OLD'}
              </text>
              {/* recovery bar */}
              <rect x={barX0} y={yc - 7} width={barW} height={14} rx={7} fill="#eceadf" />
              <rect x={barX0} y={yc - 7} width={fdW} height={14} rx={7} fill={followsNew ? COLORS.green : COLORS.orange} />
              {r.ci && (
                <g stroke="var(--ink)" strokeWidth={1.1} opacity={0.5}>
                  <line x1={barX0 + Math.max(0, Math.min(r.ci[0], 1)) * barW} x2={barX0 + Math.max(0, Math.min(r.ci[1], 1)) * barW} y1={yc} y2={yc} />
                </g>
              )}
              <text x={barX0 + barW + 8} y={yc + 4} style={{ fontFamily: 'var(--sans)', fontSize: 12, fontWeight: 700 }} fill="var(--ink)">
                {fmt(r.rec, r.byDef ? 0 : 3)}
              </text>
            </g>
          )
        })}
      </ChartSvg>

      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', lineHeight: 1.6, marginTop: 4 }}>
        On {m.label}: refreshing the field&rsquo;s own KV recovers{' '}
        <b>{fmt(m.field_only.mean, 3)}</b> of the decision&rsquo;s flip — essentially nothing —
        while keeping the field <em>stale</em> and recomputing the downstream recovers{' '}
        <b>{fmt(m.full_downstream.mean, 3)}</b>. The fresh field is ignored; the conclusion lives in
        the downstream notes. (stale and oracle are 0 and 1 by definition.)
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
        sub="Four cache states compared side by side; pick a model. Recovery numbers are the released causal-patching records."
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
        recorded the answer before any decoding starts (per-layer curves in §12, where we also pin
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
        the field. The paper names this <b>attention-mediated memoized inference</b>. This is the
        single fact behind everything in Part I — composing (§2) reuses the notes, editing (§3)
        amends them — and the keystone (§9) confirms they are one object. For the deeper evidence,
        §12–§13 stress-test the account down to the level of individual heads and features.
      </Aside>
    </Section>
  )
}
