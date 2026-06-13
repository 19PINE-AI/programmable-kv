import { useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg } from '../components/ui/Controls'
import { BarsV } from '../components/charts/BarCI'
import { ChartSvg, COLORS } from '../components/charts/core'
import { fmt, fmtMs, fmtPct, fmtX } from '../lib/format'
import systems from '../data/systems.json'

const META = { id: 'systems', num: '10', title: 'Systems payoff: a real online serving benchmark' }

function rateLabel(r: number) {
  return r === 0 ? 'saturation' : `${r} req/s`
}

function ServingDashboard() {
  const rows = systems.rows as any[]
  const [idx, setIdx] = useState(rows.length - 1)
  const row = rows[idx]

  const pcts: ('p50' | 'p90' | 'p99')[] = ['p50', 'p90', 'p99']
  const maxT = Math.max(...rows.flatMap((r) => pcts.map((p) => r.baseline.ttft_ms[p])))
  const logmax = Math.log10(maxT * 1.3)
  const logmin = Math.log10(20)
  const W = 660

  function bar(y: number, v: number, color: string, label: string) {
    const w = ((Math.log10(Math.max(v, 21)) - logmin) / (logmax - logmin)) * (W - 240)
    return (
      <g transform={`translate(170,${y})`}>
        <text x={-8} y={11} textAnchor="end" className="tick-label" style={{ fontSize: 11 }}>{label}</text>
        <rect width={Math.max(w, 2)} height={15} rx={3} fill={color} style={{ transition: 'width .4s' }} />
        <text x={Math.max(w, 2) + 7} y={12} className="tick-label" style={{ fontWeight: 700, fill: 'var(--ink)' }}>
          {fmtMs(v)}
        </text>
      </g>
    )
  }

  return (
    <div>
      <Controls>
        <ControlGroup label="offered load (Poisson arrivals)">
          <Seg
            options={rows.map((_, i) => String(i)) as any}
            value={String(idx)}
            onChange={(v) => setIdx(parseInt(v as string))}
            labels={Object.fromEntries(rows.map((r, i) => [String(i), rateLabel(r.rate)])) as any}
            accent="orange"
          />
        </ControlGroup>
      </Controls>

      <ChartSvg width={W} height={262}>
        <text x={0} y={16} style={{ fontFamily: 'var(--sans)', fontSize: 12, fontWeight: 700 }} fill={COLORS.red}>
          write the new value INTO the prefix (invalidates downstream cache blocks)
        </text>
        {bar(26, row.baseline.ttft_ms.p50, COLORS.red, 'TTFT p50')}
        {bar(46, row.baseline.ttft_ms.p90, COLORS.red, 'TTFT p90')}
        {bar(66, row.baseline.ttft_ms.p99, COLORS.red, 'TTFT p99')}

        <text x={0} y={116} style={{ fontFamily: 'var(--sans)', fontSize: 12, fontWeight: 700 }} fill={COLORS.green}>
          append-only erratum (prefix stays cache-aligned)
        </text>
        {bar(126, row.erratum.ttft_ms.p50, COLORS.green, 'TTFT p50')}
        {bar(146, row.erratum.ttft_ms.p90, COLORS.green, 'TTFT p90')}
        {bar(166, row.erratum.ttft_ms.p99, COLORS.green, 'TTFT p99')}

        <text x={0} y={216} className="tick-label">log scale →</text>

        {/* summary cards */}
        <g transform="translate(0,228)">
          {[
            { t: 'APC hit-rate', v: `${fmtPct(row.erratum.prefix_hit_rate, 1)} vs ${fmtPct(row.baseline.prefix_hit_rate, 1)}` },
            { t: 'throughput', v: `${fmt(row.erratum.throughput_req_s, 2)} vs ${fmt(row.baseline.throughput_req_s, 2)} req/s` },
            { t: 'p90 TTFT speedup', v: fmtX(row.ttft_p90_speedup, 0) },
            { t: 'throughput speedup', v: fmtX(row.throughput_speedup, 1) },
          ].map((c, i) => (
            <g key={c.t} transform={`translate(${i * 165},0)`}>
              <text className="tick-label" style={{ fontSize: 10 }}>{c.t}</text>
              <text y={17} style={{ fontFamily: 'var(--sans)', fontSize: 13.5, fontWeight: 700, fill: 'var(--ink)' }}>{c.v}</text>
            </g>
          ))}
        </g>
      </ChartSvg>
    </div>
  )
}

function SpeedupVsLoad() {
  const rows = systems.rows as any[]
  return (
    <BarsV
      groups={rows.map((r) => ({
        label: rateLabel(r.rate),
        values: [{ v: r.throughput_speedup }],
      }))}
      seriesLabels={['throughput speedup, erratum vs. in-prefix edit']}
      colors={[COLORS.orange]}
      yLabel="speedup (×)"
      yFmt={(v) => `${v.toFixed(1)}×`}
      height={250}
    />
  )
}

function VisionTtft() {
  const vt = systems.vision_ttft as any[]
  return (
    <BarsV
      groups={vt.map((v) => ({
        label: `${v.px}px (${v.img_tokens} tok)`,
        values: [{ v: v.full_ms }, { v: v.reuse_ms }],
      }))}
      seriesLabels={['full re-encode (vision tower + image prefill)', 'cached image-KV reuse']}
      colors={[COLORS.red, COLORS.green]}
      yLabel="TTFT (ms)"
      yFmt={(v) => `${v.toFixed(0)}`}
      height={260}
    />
  )
}

export function Systems() {
  const sat = (systems.rows as any[])[systems.rows.length - 1]
  return (
    <Section meta={META}>
      <P>
        Mechanism and capability only matter if they survive a real serving stack. The benchmark:
        vLLM&rsquo;s V1 engine as a genuine online server — <code>AsyncLLMEngine</code>, CUDA
        graphs, continuous batching, automatic prefix caching (APC), and <em>Poisson</em> request
        arrivals at controlled offered load — over a shared ~{((systems.prompt_tokens as number) / 1000).toFixed(0)}k-token
        agent policy ({systems.model}) with one mutable field. Two ways to apply the same field
        change:
      </P>

      <Figure
        label="Online serving."
        title="One mutable field, two ways to update it"
        sub="vLLM V1, 96 requests per arm, TTFT percentiles from client-side timestamps, APC hit-rate from the engine's own Prometheus counters"
        caption={
          <>
            Writing the new value into the prefix changes a cached block&rsquo;s content hash and
            invalidates every downstream block — the server becomes prefill-bound and saturates at
            ≈1.5 req/s, so p90 TTFT collapses under load (22–55 s). The append-only erratum keeps
            the prefix a cache hit ({fmtPct(sat.erratum.prefix_hit_rate, 1)} vs.{' '}
            {fmtPct(sat.baseline.prefix_hit_rate, 1)} APC hit-rate) and stays at 86 ms–1 s.
          </>
        }
      >
        <ServingDashboard />
      </Figure>

      <Figure
        narrow
        label="The advantage grows with load."
        caption={
          <>
            Exactly as predicted for a compute-bound vs. cache-bound regime: {fmtX(1.6, 1)} at 2
            req/s rising to <b>{fmtX(sat.throughput_speedup, 1)}</b> at saturation. This is the
            serving translation of §6&rsquo;s append-only property — the edit is not just correct,
            it is <em>cache-shaped</em>.
          </>
        }
      >
        <SpeedupVsLoad />
      </Figure>

      <H3>It also holds in a real agent environment</H3>
      <P>
        On the τ²-bench retail environment — single tool-decisions and a multi-turn autonomous
        loop scored by the environment&rsquo;s own tool enforcement — an agent that reuses a stale
        cache after a state change collapses, while <code>field+erratum</code> preserves task
        success at a fraction of the recompute. On the real ~1.4k-token retail policy, transplant
        reproduces the clean decision; the one hard case (a long, buried field that must flip a
        conclusion) needs the robust <code>field+erratum</code> edit — the same long-context
        lesson §6 predicts, now in a real environment.
      </P>

      <Figure
        narrow
        label="The multimodal serving win."
        caption={
          <>
            Reusing a cached image (skipping the vision tower and image-token prefill entirely)
            accelerates first-token latency 2.4–8.4×, growing with image size (Qwen2.5-VL-7B).
          </>
        }
      >
        <VisionTtft />
      </Figure>

      <Aside>
        <b>Why the in-prefix baseline is the right one.</b> It is what every serving stack does
        today when a templated field changes: re-render the prompt, lose the prefix match. The
        erratum is behaviorally equivalent (§6) and turns the same update into an append — the
        entire 53–398× p90 gap is downstream of that one representational choice.
      </Aside>
    </Section>
  )
}
