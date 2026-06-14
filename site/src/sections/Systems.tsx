import { useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg } from '../components/ui/Controls'
import { BarsV } from '../components/charts/BarCI'
import { ChartSvg, COLORS } from '../components/charts/core'
import { fmt, fmtMs, fmtPct, fmtX } from '../lib/format'
import systems from '../data/systems.json'

const META = { id: 'systems', num: '5', title: "Why it's faster" }

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
        <ControlGroup label="how busy the server is (requests arriving)">
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
          rewrite the saved prompt (the server must redo its work)
        </text>
        {bar(26, row.baseline.ttft_ms.p50, COLORS.red, 'typical reply')}
        {bar(46, row.baseline.ttft_ms.p90, COLORS.red, 'slow-case reply')}
        {bar(66, row.baseline.ttft_ms.p99, COLORS.red, 'worst-case reply')}

        <text x={0} y={116} style={{ fontFamily: 'var(--sans)', fontSize: 12, fontWeight: 700 }} fill={COLORS.green}>
          add a short correction note (the saved work stays valid)
        </text>
        {bar(126, row.erratum.ttft_ms.p50, COLORS.green, 'typical reply')}
        {bar(146, row.erratum.ttft_ms.p90, COLORS.green, 'slow-case reply')}
        {bar(166, row.erratum.ttft_ms.p99, COLORS.green, 'worst-case reply')}

        <text x={0} y={216} className="tick-label">time to first reply (log scale) →</text>

        {/* summary cards */}
        <g transform="translate(0,228)">
          {[
            { t: 'reused saved work', v: `${fmtPct(row.erratum.prefix_hit_rate, 1)} vs ${fmtPct(row.baseline.prefix_hit_rate, 1)}` },
            { t: 'requests handled at once', v: `${fmt(row.erratum.throughput_req_s, 2)} vs ${fmt(row.baseline.throughput_req_s, 2)} req/s` },
            { t: 'slow-case reply, faster by', v: fmtX(row.ttft_p90_speedup, 0) },
            { t: 'requests handled, more by', v: fmtX(row.throughput_speedup, 1) },
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
      seriesLabels={['how many more requests it handles: correction note vs. rewriting the prompt']}
      colors={[COLORS.orange]}
      yLabel="more requests handled (×)"
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
      seriesLabels={['re-process the whole image', 'reuse the saved image notes']}
      colors={[COLORS.red, COLORS.green]}
      yLabel="time to first reply (ms)"
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
        Here is the real-world payoff. As the model reads a prompt, it saves its work in a kind of
        notebook of notes (the &ldquo;cache&rdquo;) so it does not have to redo that thinking. We ran
        a real production-style server (vLLM, a popular open-source serving engine) sending it a
        steady stream of requests, all sharing the same ~{((systems.prompt_tokens as number) / 1000).toFixed(0)}k-word
        instruction document ({systems.model}) with one detail that can change. There are two ways
        to make that change:
      </P>

      <Figure
        label="On a live server."
        title="One changed detail, two ways to update it"
        sub="A live server, 96 requests each way. Response times measured at the client; how often saved work is reused, read straight from the server's own counters."
        caption={
          <>
            Rewriting the saved prompt forces the server to redo its work for every request. It
            falls behind, handling only about 1.5 requests per second, and the slow-case response
            time (the 90th percentile) balloons to 22&ndash;55 seconds. Adding a short correction
            note instead keeps the saved work valid &mdash; reused{' '}
            {fmtPct(sat.erratum.prefix_hit_rate, 1)} of the time vs.{' '}
            {fmtPct(sat.baseline.prefix_hit_rate, 1)} &mdash; and replies stay fast, 86 ms to 1 s.
          </>
        }
      >
        <ServingDashboard />
      </Figure>

      <Figure
        narrow
        label="The busier the server, the bigger the win."
        caption={
          <>
            When traffic is light the lead is modest ({fmtX(1.6, 1)} more requests handled at 2
            requests per second), but as the server gets busier it grows to{' '}
            <b>{fmtX(sat.throughput_speedup, 1)}</b>. The reason is simple: the correction note
            keeps the saved work reusable, so the server never has to redo it.
          </>
        }
      >
        <SpeedupVsLoad />
      </Figure>

      <H3>It also works for a real customer-support agent</H3>
      <P>
        We also tried it on a realistic customer-support agent benchmark (a retail help desk),
        where the agent has to make the right calls and follow the rules to finish a task. If the
        agent keeps reusing its saved notes after something changes, it gets the wrong answer. The
        correction-note approach (<code>field+erratum</code>) keeps the agent succeeding while
        skipping most of the redo work. On the real ~1.4k-word retail policy, simply copying over
        the saved notes reproduces the clean decision; the one tricky case &mdash; a detail buried
        deep in a long document that has to flip the final answer &mdash; needs the sturdier
        correction-note edit. That is the same lesson the mechanism predicts, now in a real setting.
      </P>

      <Figure
        narrow
        label="The same win for images."
        caption={
          <>
            Reusing the saved notes for an image, instead of processing it from scratch, makes the
            first reply 2.4&ndash;8.4× faster &mdash; and the bigger the image, the bigger the
            gain (Qwen2.5-VL-7B).
          </>
        }
      >
        <VisionTtft />
      </Figure>

      <Aside>
        <b>Why compare against rewriting the prompt?</b> Because that is what servers do today when
        a detail changes: they rebuild the prompt and throw away the saved work. The correction
        note gives the same answer (we showed this earlier) but turns the update into a tiny add-on
        &mdash; and that single choice is what produces the whole 53&ndash;398× gap in slow-case
        response time.
      </Aside>
    </Section>
  )
}
