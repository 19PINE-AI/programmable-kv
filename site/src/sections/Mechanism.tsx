import { useMemo, useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH } from '../components/charts/BarCI'
import { ChartSvg, COLORS, Legend } from '../components/charts/core'
import { fmt, fmtPct } from '../lib/format'
import mechanism from '../data/mechanism.json'

const META = { id: 'mechanism', num: '7', title: 'Models take notes while reading' }

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
        On {m.label}: updating the fact alone moves the answer by{' '}
        <b>{fmt(m.field_only.mean, 3)}</b> — essentially nothing —
        while leaving the fact <em>untouched</em> and refreshing the notes taken after it moves it by{' '}
        <b>{fmt(m.full_downstream.mean, 3)}</b>. The fresh fact is ignored; the answer was already
        written into the notes. (Doing nothing scores 0; rereading the whole new prompt scores 1.)
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
      xLabel="how much the answer moved"
      refX={[{ x: 0, label: 'no change' }, { x: 1, label: 'fully' }]}
      markerLabel="when the after-the-fact notes are refreshed"
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
            id: 'suffix', label: 'refresh notes after the fact (most important first)', color: COLORS.orange,
            points: (m.cum_suffix as any[]).map((p) => ({ x: p.frac, y: p.mean, lo: p.ci?.[0], hi: p.ci?.[1] })), band: true,
          },
          {
            id: 'prefix', label: 'refresh notes before the fact (comparison)', color: COLORS.gray, dash: true,
            points: (m.cum_prefix as any[]).map((p) => ({ x: p.frac, y: p.mean, lo: p.ci?.[0], hi: p.ci?.[1] })),
          },
        ]}
        xLabel="share of notes refreshed"
        yLabel="how much the answer moved"
        yDomain={[-0.08, 1.08]}
        xFmt={(v) => fmtPct(v, 0)}
        refLinesY={[{ y: 1, label: 'oracle' }, { y: 0, label: 'stale' }]}
        highlightX={frac}
        height={300}
      />
      <Legend items={[
        { label: 'notes after the fact', color: COLORS.orange },
        { label: 'notes before the fact — comparison', color: COLORS.gray, dash: true },
      ]} />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        With {fmtPct(frac, 0)} of the notes refreshed: refreshing notes taken after the fact moves the answer by{' '}
        <b>{fmt(nearest.mean, 2)}</b>; refreshing the same share of notes taken before it moves it by{' '}
        <b>{fmt(nearestPre.mean, 2)}</b>. The answer lives in the notes that come after the fact,
        spread across many of them.
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
      xLabel="how much fixing the fact alone moves the answer (Qwen3-8B)"
      refX={[{ x: 1, label: 'fully' }]}
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
      xLabel={`chance of the safe answer — corrected fact in the prompt, plus one added line (Qwen3-8B, n=${mechanism.wording_n})`}
      labelWidth={215}
    />
  )
}

export function Mechanism() {
  const q8 = (mechanism.models as any[]).find((m) => m.tag === 'qwen3_8b')!
  return (
    <Section meta={META}>
      <P>
        Here is the surprising part. When a model reads a prompt, it keeps a running set of notes —
        a kind of scratchpad it builds up as it goes (researchers call it the &ldquo;KV cache&rdquo;).
        You might assume those notes just hold the words it read. They hold more than that. At certain
        points after an important fact — think of the moment after a sentence ends or a section
        breaks, little summary spots where it pauses to take stock — the model has already worked out
        the answer and written that conclusion into its notes. Later, when it replies, it reads back
        those notes, not the original fact. So if you correct the fact but leave the notes alone,
        nothing changes: the old answer is sitting elsewhere. We tested this by surgically swapping
        pieces of the notes and watching the answer follow. Try the four versions below — each one
        keeps some parts old and refreshes others:
      </P>

      <Figure
        label="Test 1 — where the answer lives."
        title="Swap pieces of the notes, watch the answer"
        sub="Four versions compared side by side; pick a model. The numbers are from our released measurements."
        caption={
          <>
            Refreshing the model&rsquo;s notes about the fact itself changes the answer almost not at
            all (between −0.028 and 0.14 across eleven models); leaving the fact untouched and
            refreshing the notes taken after it changes the answer completely. The fact is read only
            indirectly — it drives under 1% of the decision.
          </>
        }
      >
        <CachePatchLab />
      </Figure>

      <Figure
        narrow
        label="Every model we tested."
        caption={
          <>
            Refreshing the fact alone (orange bars, with error ranges) versus refreshing the notes
            after it (orange diamonds, near 1.0) for every model in the study — four families of
            models, from small to large. The pattern holds across all of them.
          </>
        }
      >
        <FieldVsDownstreamStrip />
      </Figure>

      <H3>Test 2 — the answer is spread across the notes after the fact</H3>
      <P>
        Which notes hold the old answer? We refreshed a growing share of them — starting with the
        ones that matter most — and watched how much the answer moved. Drag the slider:
      </P>
      <Figure
        label="Test 2 — the answer sits after the fact."
        caption={
          <>
            The answer moves only as we refresh notes taken after the fact, and is fully recovered by
            the time we have refreshed almost all of them; refreshing the same share of notes taken
            before the fact (a comparison case) does nothing. On Qwen3-8B, refreshing just the top 2%
            of the after-the-fact notes already moves the answer by{' '}
            {fmt((q8.cum_suffix as any[])[0].mean, 2)}.
          </>
        }
      >
        <SuffixConcentration />
      </Figure>

      <H3>Test 3 — the answer is actually written down, not just inferable</H3>
      <P>
        We can read the model&rsquo;s notes directly. A simple reader trained on them recovers the
        answer from the after-the-fact notes the moment the model finishes reading — before it has
        said a single word. So the answer is genuinely recorded, not figured out on the fly (the
        full breakdown, including exactly <em>when</em> the note gets written, is in §12). The chart
        below shows the catch depends on <em>where</em> the fact sits: the later it appears, the
        fewer notes come after it to capture an answer — so correcting the fact in place works
        better.
      </P>
      <Figure
        narrow
        label="Test 4 — it depends on where the fact sits."
        caption={
          <>
            How much a fix to the fact alone moves the answer, as the fact appears later in the
            prompt (Qwen3-8B). Move it all the way to the end — the common practical workaround —
            and nothing comes after it, no answer gets recorded, and editing it in place just works.
            That popular workaround turns out to be a special case of what is going on here.
          </>
        }
      >
        <DoseResponse />
      </Figure>

      <H3>What the note actually holds</H3>
      <P>
        If the note were just a copy of the fact, adding a line of text afterward would do nothing.
        So we tested exactly that. Each version starts from a clean prompt that already has the
        corrected fact, then adds one line at the end. Restating the value, or labeling it as an
        update, changes nothing. But bluntly telling the model that its{' '}
        <em>&ldquo;earlier conclusion is void — re-evaluate from scratch&rdquo;</em> actively
        backfires:
      </P>
      <Figure
        narrow
        label="Trying different wording."
        caption={
          <>
            Chance of the safe answer, with the corrected fact in the prompt plus one added line
            (Qwen3-8B). Adding nothing, restating the value, and labeling it an update all sit at
            1.0; a plain explicit override is no different; the confrontational
            &ldquo;re-evaluate&rdquo; phrasing drops to{' '}
            {fmt((mechanism.wording as any[]).find((x) => x.key === 'conclusion')!.P_safe, 2)}.
            The note behaves like an <b>answer the model has already settled on</b>: a clear,
            late correction can overwrite it, but trying to argue it down only throws the answer
            off balance.
          </>
        }
      >
        <WordingAblation />
      </Figure>

      <Aside>
        <b>The one idea.</b> While the model reads, it works out the answer to the fact and writes
        that answer into its notes — at summary spots that come after the fact. When it later
        replies, it reads the answer from those notes, not from the fact itself. The paper calls
        this <b>attention-mediated memoized inference</b>. It is the single idea behind everything
        in Part I — reusing notes lets you combine them (§2), and amending notes lets you edit them
        (§3) — and §9 confirms they are all the same thing. For the deeper evidence, §12–§13 take
        the explanation apart down to individual pieces of the model.
      </Aside>
    </Section>
  )
}
