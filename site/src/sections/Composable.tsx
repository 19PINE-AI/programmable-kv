import { useState } from 'react'
import { Section, P, H3, Aside, PaperConst } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH } from '../components/charts/BarCI'
import { COLORS, Legend } from '../components/charts/core'
import { RopeRotation } from '../components/diagrams/RopeRotation'
import { fmt, fmtX, fmtTokens } from '../lib/format'
import composing from '../data/composing.json'
import constants from '../data/constants.json'

const META = { id: 'composable', num: '2', title: 'Load a skill once: the composable cache' }

function TtftScaling() {
  const sc = composing.scaling as any[]
  const [tag, setTag] = useState('qwen3_8b')
  const m = sc.find((x) => x.tag === tag)!
  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={sc.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
      </Controls>
      <LineChart
        series={[
          { id: 'full', color: COLORS.red, points: m.points.map((p: any) => ({ x: p.L, y: p.full_ms })) },
          { id: 'pre', color: COLORS.blue, points: m.points.map((p: any) => ({ x: p.L, y: p.precomp_ms })) },
        ]}
        xLog
        yLog
        xTicks={m.points.map((p: any) => p.L)}
        xFmt={(v) => fmtTokens(v)}
        yFmt={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}s` : `${v.toFixed(0)}ms`)}
        xLabel="skill length L (tokens)"
        yLabel="time to first token"
        yDomain={[Math.min(...m.points.map((p: any) => p.precomp_ms)) * 0.7, Math.max(...m.points.map((p: any) => p.full_ms)) * 1.4]}
        height={300}
      />
      <Legend items={[
        { label: 'full reprefill — O(L²)', color: COLORS.red },
        { label: 'transplant (re-rotate + splice) — O(L)', color: COLORS.blue },
      ]} />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        speedups on {m.label}:{' '}
        {m.points.map((p: any) => `${fmtX(p.speedup, 1)} @ ${fmtTokens(p.L)}`).join(' · ')}
      </div>
    </div>
  )
}

function DomainScorecard() {
  const dom = composing.domains as any[]
  const [tag, setTag] = useState(dom[0]?.tag ?? 'llama31_8b')
  const m = dom.find((x) => x.tag === tag)!
  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={dom.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
        {m.summary && (
          <span style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)' }}>
            transplanted ≡ full: <b>{m.summary.agree[0]}/{m.summary.agree[1]} domains</b> (mean cos {fmt(m.summary.cos, 3)})
          </span>
        )}
      </Controls>
      <table className="data-table">
        <thead>
          <tr><th>skill domain</th><th>skill tokens</th><th>full reprefill</th><th>transplanted</th><th>logit cos</th><th>same decision</th></tr>
        </thead>
        <tbody>
          {m.rows.map((r: any) => (
            <tr key={r.domain}>
              <td style={{ fontWeight: 600 }}>{r.domain}</td>
              <td>{r.skill_tok}</td>
              <td>{r.full}</td>
              <td>{r.reposition}</td>
              <td>{fmt(r.cos, 2)}</td>
              <td style={{ color: r.agree ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>{r.agree ? '✓' : '✗'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, color: 'var(--ink-faint)', marginTop: 6 }}>
        parsed directly from the released run logs (<code>comp_div_*.log</code>); domains span refund
        policy, access control, deployment gates, prescriptions, loans, legal holds, incident
        response, and visa rules
      </div>
    </div>
  )
}

function FidelityBar() {
  const rows = constants.transplant_cosine_bar.rows
  return (
    <BarsH
      items={rows.map((r: any) => ({ label: r.label, value: r.cos, color: COLORS.blue }))}
      domain={[0.85, 1.01]}
      xLabel="next-token logit cosine: transplanted skill vs. full recompute"
      refX={[{ x: 1.0, label: 'identical', color: COLORS.green }]}
      valueFmt={(v) => v.toFixed(3)}
      labelWidth={180}
    />
  )
}

function Agentic() {
  const ag = composing.agentic as any[]
  return (
    <BarsH
      items={ag.map((m) => ({
        label: m.label,
        value: m.agreement,
        lo: m.agreement_ci?.[0],
        hi: m.agreement_ci?.[1],
        color: COLORS.purple,
      }))}
      domain={[0.25, 1.05]}
      xLabel="transplanted-vs-full tool-call agreement (N=108 actual function calls, bootstrap CIs)"
      refX={[{ x: 1, label: 'identical' }]}
      labelWidth={185}
    />
  )
}

const MULTI_COLORS = [COLORS.blue, COLORS.orange, COLORS.green, COLORS.purple, COLORS.red, COLORS.gray]

function MultiSkill() {
  const mu = composing.multi as any[]
  return (
    <div>
      <LineChart
        series={mu.map((m, i) => ({
          id: m.tag,
          color: MULTI_COLORS[i % 6],
          points: m.points.map((p: any) => ({ x: p.N, y: p.cos })),
          marker: false,
        }))}
        xTicks={[1, 2, 3, 4]}
        xLabel="number of independently-precompiled skills spliced into one context"
        yLabel="logit cosine vs. full recompute"
        yDomain={[0.5, 1.02]}
        height={260}
      />
      <Legend items={mu.map((m, i) => ({ label: m.label, color: MULTI_COLORS[i % 6] }))} />
    </div>
  )
}

export function Composable() {
  return (
    <Section meta={META}>
      <P>
        The first challenge was <strong>loading skills</strong>: a reusable skill — a policy, a
        tool spec — should be loadable once and reused anywhere, not re-prefilled every time it
        lands in a new context. It can be. Compute the skill&rsquo;s KV once, in isolation, then
        move it to wherever it sits in a new context. Because attention libraries cache{' '}
        <em>post-RoPE</em> keys, moving a chunk means re-rotating its keys to the target positions
        (values carry no position at all). The reason this preserves behavior — a skill&rsquo;s notes
        are re-derivable from context the decision can still see — is the mechanism we unpack in §7.
      </P>

      <Figure
        label="Position-portable transplant."
        title="RoPE is a rotation, so repositioning is a re-rotation"
        caption={
          <>
            The skill&rsquo;s keys precompiled at positions 0…L−1 are rotated forward by Δpos and
            spliced into the live cache; the values copy verbatim. The caching machinery here is
            prior art (Prompt Cache, CacheBlend, EPIC, CacheSlide) — the paper&rsquo;s contribution
            is the mechanism that predicts <em>when this preserves behavior</em>, and the
            decision-governance evaluation showing it does.
          </>
        }
      >
        <RopeRotation />
      </Figure>

      <Figure
        label="Linear-time TTFT."
        caption={
          <>
            Full reprefill of a length-L skill is O(L²); transplant is O(L) — a re-rotation pass
            plus the suffix. The speedup grows with skill length: 13.9× at 32k tokens on an 8B
            model.
          </>
        }
      >
        <TtftScaling />
      </Figure>

      <H3>But does the transplanted skill still govern behavior?</H3>
      <P>
        Throughput is the easy half. The paper&rsquo;s evaluation lens is{' '}
        <strong>decision governance</strong>: after the splice, does the skill still control the
        model&rsquo;s tool decision, exactly as a fresh prefill would?
      </P>

      <Figure
        narrow
        label="Eight domains, per-model."
        caption={
          <>
            A precompiled, repositioned skill reproduces the full-reprefill decision domain by
            domain. The one residual error across the study is a <em>seam</em> at the
            chunk&rsquo;s start — the few tokens that would have attended to the now-missing
            prefix — and recomputing 1–2 boundary tokens closes it. §7&rsquo;s mechanism is why it
            is the boundary, specifically, that needs repair.
          </>
        }
      >
        <DomainScorecard />
      </Figure>

      <Figure
        narrow
        label="Transplant fidelity across the model family."
        caption={
          <>
            Next-token logit cosine between the spliced skill and a full reprefill, across
            families and scales including FP8 and MoE checkpoints.{' '}
            <PaperConst src={constants.transplant_cosine_bar.source} />
          </>
        }
      >
        <FidelityBar />
      </Figure>

      <Figure
        narrow
        label="Agentic tool-calling, measured with actual function calls."
        caption={
          <>
            Transplant preserves real tool-calling — not a proxy metric — across 8 models. The one
            consistent exception in the wider study is sliding-window attention (Gemma), diagnosed
            and fixed in §11.
          </>
        }
      >
        <Agentic />
      </Figure>

      <Figure
        narrow
        label="A library of skills composes."
        caption={
          <>
            Logit cosine to full recompute as N=1–4 independently-precompiled skills are spliced
            into one context. The recorded decision matches full recompute in nearly every cell;
            the exceptions sit at cosine ≥0.996 — boundary flips of a near-tied decision, the same
            sensitivity quantified in §10, not transplant damage.
          </>
        }
      >
        <MultiSkill />
      </Figure>

      <Aside>
        <b>Why it works, in one line.</b> A skill&rsquo;s notes memoize conclusions about{' '}
        <em>the skill&rsquo;s own content</em>; as long as the decision can still see the live
        context it needs, the only thing position changes is the keys&rsquo; rotation — which is
        exactly invertible.
      </Aside>
    </Section>
  )
}
