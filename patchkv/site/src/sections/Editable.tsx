import { useState } from 'react'
import { scaleLog, scaleLinear } from 'd3-scale'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg, ModelPicker } from '../components/ui/Controls'
import { Heatmap, ramp } from '../components/charts/Heatmap'
import { BarsH } from '../components/charts/BarCI'
import { AxisBottom, AxisLeft, ChartSvg, COLORS } from '../components/charts/core'
import { fmt, fmtPct, fmtMs } from '../lib/format'
import editing from '../data/editing.json'
import prompts from '../data/prompts.json'

const META = { id: 'editable', num: '6', title: 'Consequence I: the cache is editable' }

const METHOD_INFO: Record<string, { label: string; desc: string; color: string }> = {
  full_reprefill: { label: 'full reprefill', color: COLORS.gray, desc: 'recompute the whole context — correct but pays the full quadratic prefill' },
  hoist_to_end: { label: 'hoist-to-end', color: COLORS.purple, desc: 'rewrite the prompt so the mutable field sits at the end — cheapest, but demands prompt surgery and pre-identifying every mutable field' },
  'field+erratum': { label: 'field + erratum', color: COLORS.green, desc: 'refresh the field KV and append a one-line salient erratum — the paper’s robust default; no prompt surgery' },
  erratum: { label: 'erratum only', color: COLORS.green, desc: 'append the one-line erratum, leave everything stale — the notes are amended, not recomputed' },
  'cacheblend@15%': { label: 'CacheBlend@15%', color: COLORS.blue, desc: 'KV-deviation-ranked selective recompute — chases changed keys rather than the tokens that memoized the conclusion, and fails here' },
  in_place: { label: 'in-place edit', color: COLORS.orange, desc: 'refresh only the field’s KV (~0.6% recompute) — near-free, but recovers nothing without reasoning' },
  stale: { label: 'stale (reuse)', color: COLORS.red, desc: 'reuse everything — the do-nothing baseline' },
}

function Frontier() {
  const methods = (editing.baseline.methods as any[]).filter((m) => METHOD_INFO[m.method])
  const [sel, setSel] = useState('field+erratum')
  const W = 660
  const H = 320
  const pad = { l: 56, r: 20, t: 16, b: 46 }
  const x = scaleLog().domain([0.004, 1.4]).range([pad.l, W - pad.r])
  const y = scaleLinear().domain([-0.04, 1.06]).range([H - pad.b, pad.t])
  const cur = methods.find((m) => m.method === sel)!
  const info = METHOD_INFO[sel]

  return (
    <div>
      <Controls>
        <ControlGroup label="method">
          <Seg
            options={methods.map((m) => m.method) as any}
            value={sel}
            onChange={(v) => setSel(v as string)}
            labels={Object.fromEntries(methods.map((m) => [m.method, METHOD_INFO[m.method].label])) as any}
          />
        </ControlGroup>
      </Controls>

      <ChartSvg width={W} height={H}>
        <AxisBottom scale={x as any} y={H - pad.b} ticks={[0.005, 0.01, 0.05, 0.1, 0.5, 1]} fmt={(v) => fmtPct(v, v < 0.01 ? 1 : 0)} label="fraction of the context recomputed (log scale)" grid gridY1={pad.t} gridY2={H - pad.b} />
        <AxisLeft scale={y as any} x={pad.l} ticks={[0, 0.25, 0.5, 0.75, 1]} label="P(correct decision)" grid gridX2={W - pad.r} />
        {methods.map((m) => {
          const fr = Math.max(m.recompute_frac, 0.005)
          const isSel = m.method === sel
          return (
            <g key={m.method} onClick={() => setSel(m.method)} style={{ cursor: 'pointer' }}>
              <circle cx={x(fr)} cy={y(m.P_correct)} r={isSel ? 10 : 7}
                fill={METHOD_INFO[m.method].color} opacity={isSel ? 1 : 0.7}
                stroke={isSel ? 'var(--ink)' : '#fff'} strokeWidth={isSel ? 2 : 1.2} />
              <text className="tick-label" x={x(fr)} y={y(m.P_correct) + (m.P_correct > 0.5 ? 24 : -14)}
                textAnchor="middle" style={{ fontWeight: isSel ? 700 : 400, fill: isSel ? 'var(--ink)' : undefined }}>
                {METHOD_INFO[m.method].label}
              </text>
            </g>
          )
        })}
      </ChartSvg>

      <div className="aside" style={{ marginTop: 4 }}>
        <b>{info.label}</b> — P(correct) {fmt(cur.P_correct, 2)} at {fmtPct(cur.recompute_frac, 1)} recompute.{' '}
        {info.desc}
        {(sel === 'erratum' || sel === 'field+erratum') && (
          <div className="mono" style={{ background: '#fff', borderRadius: 4, padding: '6px 10px', marginTop: 8, fontSize: 11.5 }}>
            {prompts.scenarios[0].erratum_line}
          </div>
        )}
      </div>
    </div>
  )
}

function KsweepHeat() {
  const ks = editing.ksweep as any[]
  const Ks = ks[0].Ks.map((x: any) => x.K)
  return (
    <div>
      <Heatmap
        rows={ks.map((m) => m.label)}
        cols={Ks.map((k: number) => (k === 0 ? 'field only' : `+top ${k}`))}
        value={(r, c) => ks[r].Ks[c]?.P_correct ?? null}
        colorOf={(v) => (v === null ? '#f0eee6' : ramp(v, [74, 138, 92]))}
        colLabel="field + K highest-effect downstream tokens recomputed (under reasoning)"
        rowLabelWidth={160}
        tooltip={(r, c) => `${ks[r].label}, K=${Ks[c]}: P(correct) ${fmt(ks[r].Ks[c].P_correct, 2)} (full=${fmt(ks[r].full, 2)})`}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        K&nbsp;★ (minimal K reaching full quality):{' '}
        {ks.filter((m) => m.K_star != null).map((m) => `${m.label} ${m.K_star}`).join(' · ')}
        {' — '}and for several models no small K suffices.
      </div>
    </div>
  )
}

function ArchBar() {
  const arch = (editing.arch as any[]).filter((a) => a.reasoning.recovery != null)
  const ARCH_LABEL: Record<string, string> = {
    attention: 'full attention (GQA)', gqa: 'full attention (GQA)',
    sliding_window: 'sliding-window', hybrid: 'hybrid attn+SSM', ssm: 'pure SSM',
  }
  return (
    <BarsH
      items={arch.map((a) => ({
        label: `${a.label} — ${ARCH_LABEL[(a.arch ?? '').toLowerCase()] ?? a.arch}`,
        value: a.reasoning.recovery,
        lo: a.reasoning.ci?.[0],
        hi: a.reasoning.ci?.[1],
        color: (a.arch ?? '').toLowerCase().includes('ssm') && !(a.arch ?? '').toLowerCase().includes('hybrid')
          ? COLORS.red
          : (a.arch ?? '').toLowerCase() === 'hybrid' ? COLORS.purple : COLORS.green,
      }))}
      domain={[0, 1.1]}
      xLabel="erratum recovery under reasoning"
      refX={[{ x: 1, label: 'oracle' }]}
      labelWidth={290}
    />
  )
}

function WeightEditTable() {
  const w = editing.weight as any
  const rows: { key: string; label: string }[] = [
    { key: 'kv_erratum', label: 'field+erratum (KV cache)' },
    { key: 'kv_inplace', label: 'in-place (KV cache)' },
    { key: 'rome', label: 'ROME (rank-one weight edit)' },
    { key: 'lora_ft', label: 'LoRA fine-tune (weights)' },
  ]
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>method</th><th>flips the decision?</th><th>edit latency</th>
          <th>cross-request contamination</th><th>collateral damage</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const m = w.methods[r.key]
          if (!m) return null
          const isKV = r.key.startsWith('kv')
          return (
            <tr key={r.key} style={isKV ? { background: '#f3f8f4' } : undefined}>
              <td style={{ fontWeight: 600 }}>{r.label}</td>
              <td>{m.efficacy_deny ? '✓' : '✗ (without CoT)'}</td>
              <td>
                {fmtMs(m.latency_ms)}
                {m.cov_estimate_s ? ` + ${m.cov_estimate_s.toFixed(0)} s covariance` : ''}
              </td>
              <td style={{ color: (m.isolation_contamination ?? 0) > 0 ? 'var(--red)' : 'var(--green)', fontWeight: 600 }}>
                {fmt(m.isolation_contamination ?? 0, 1)}
              </td>
              <td style={{ color: (m.collateral_drift ?? 0) > 0 ? 'var(--red)' : 'var(--green)', fontWeight: 600 }}>
                {fmt(m.collateral_drift ?? 0, 1)}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export function Editable() {
  return (
    <Section meta={META}>
      <P>
        If the stale notes carry an old conclusion, the cheapest correct intervention is not
        recomputation — it is <strong>amending the notes</strong>: append a one-line, salient{' '}
        <em>erratum</em> late in the context, where the decision token will attend to it as a
        fresh, authoritative note. The edit is append-only, so the entire cached prefix stays
        byte-identical and cache-aligned (which is what makes §10&rsquo;s serving numbers
        possible). Click through the frontier:
      </P>

      <Figure
        label="The editing frontier."
        title="Cost vs. correctness — there is no free lunch, but there is a cheap one"
        sub={`Qwen3-8B, ${editing.baseline.n_tasks} gated tasks; x log-scale`}
        caption={
          <>
            No single method dominates. Hoist-to-end is cheapest but demands prompt surgery;{' '}
            <b>field+erratum matches its oracle correctness with no surgery</b> at a one-line
            append; the in-place edit is near-free but recovers nothing without reasoning; a
            KV-deviation-ranked recompute (CacheBlend-style) fails here because it chases changed
            keys rather than the tokens that memoized the conclusion.
          </>
        }
      >
        <Frontier />
      </Figure>

      <H3>The surgical option, mapped honestly</H3>
      <P>
        §2&rsquo;s specificity result suggests a surgical alternative: recompute the field plus
        the K highest-effect downstream tokens. It works — sometimes. Under reasoning, the minimal
        K to reach full quality is wildly model-dependent, because how <em>sticky</em> the memoized
        conclusion is does not track scale:
      </P>
      <Figure
        label="field+selective@K."
        caption={
          <>
            P(correct) as K grows, per model. K★ ≈ 4 suffices at 8B while 4B needs &gt;64; without
            reasoning, no small K helps at any scale (0.00 recovery). The paper reports{' '}
            <code>field+selective</code> as a genuine but <em>unreliable</em> tool — effective when
            the conclusion is not sticky — rather than as a default.
          </>
        }
      >
        <KsweepHeat />
      </Figure>

      <Figure
        narrow
        label="The fix is an attention-architecture method."
        caption={
          <>
            Erratum recovery under reasoning across backbones: strong on full-attention and
            sliding-window models, partial on a hybrid attention+SSM model, weak on a pure SSM
            whose recurrent state has no per-token look-back. The mechanism — and therefore the
            fix — lives in attention&rsquo;s ability to re-read a late note.
          </>
        }
      >
        <ArchBar />
      </Figure>

      <H3>Why not edit the weights instead?</H3>
      <P>
        A natural objection: to act on a changed fact, why not edit the model itself — ROME, or a
        quick LoRA? The paper runs the comparison with a <em>faithful</em> ROME (validated first on
        the canonical Eiffel-Tower edit, so the baseline is not crippled). All three methods flip
        the target decision. The difference is everything else:
      </P>
      <Figure
        label="KV editing vs. weight editing (Llama-3.1-8B)."
        caption={
          <>
            A weight edit is <em>global</em>: the same model instance can no longer hold{' '}
            <code>status=shipped</code> for one request and <code>pending</code> for another —
            every concurrent order that is genuinely still pending gets wrongly flipped
            (contamination 1.0), and half of an unrelated decision battery drifts. The append-only
            erratum lives in a <em>per-sequence</em> cache: zero contamination, zero collateral,
            30–50× faster. Weight editing is for durable global facts; mutable per-request state
            is the editable cache&rsquo;s niche.
          </>
        }
      >
        <WeightEditTable />
      </Figure>

      <Aside>
        <b>Practical recipe.</b> Default to <code>field+erratum</code> (robust, append-only,
        cache-aligned). If you are running a reasoning model and the context is benign, the
        in-place edit is a free fast-path — the chain re-reads the field. Reserve{' '}
        <code>field+selective@K</code> for models you have measured.
      </Aside>
    </Section>
  )
}
