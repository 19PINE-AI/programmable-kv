import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { DiagonalScatter } from '../components/charts/DiagonalScatter'
import { BarsH } from '../components/charts/BarCI'
import { COLORS, Legend } from '../components/charts/core'
import { fmt, fmtX } from '../lib/format'
import keystone from '../data/keystone.json'

const META = { id: 'keystone', num: '9', title: 'One substrate: edit and compose are the same object' }

const METHOD_COLORS: Record<string, string> = {
  in_place: COLORS.orange,
  'sel@8': COLORS.purple,
  'sel@32': COLORS.blue,
  erratum: COLORS.green,
}

function KeystoneScatter() {
  const ce = keystone.compose_edit as any[]
  const points = ce.flatMap((m) =>
    m.methods.map((meth: any) => ({
      x: meth.recomputed,
      y: meth.composed,
      label: `${m.label} · ${meth.method}: recomputed ${fmt(meth.recomputed, 2)}, composed ${fmt(meth.composed, 2)}`,
      color: METHOD_COLORS[meth.method] ?? COLORS.gray,
      symbol: (m.tag === 'gemma2_9b' ? 'square' : 'circle') as 'square' | 'circle',
    })),
  )
  return (
    <div>
      <DiagonalScatter
        points={points}
        xLabel="edit recovery in a RECOMPUTED cache"
        yLabel="edit recovery in a COMPOSED cache"
      />
      <Legend
        items={[
          { label: 'in-place', color: COLORS.orange },
          { label: 'selective@8', color: COLORS.purple },
          { label: 'selective@32', color: COLORS.blue },
          { label: 'erratum', color: COLORS.green },
        ]}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, color: 'var(--ink-faint)', marginTop: 4 }}>
        circles = Llama-3.1-8B (n=8) · squares = Gemma-2-9B; values are recovery ratios and may exceed 1
      </div>
    </div>
  )
}

function UnifiedAgent() {
  const ag = keystone.agent as any[]
  return (
    <div>
      <BarsH
        items={ag.map((m) => ({
          label: m.label,
          value: m.agreement,
          lo: m.agreement_ci?.[0],
          hi: m.agreement_ci?.[1],
          color: COLORS.blue,
          note: `· ${fmtX(m.speedup, 1)}`,
        }))}
        domain={[0.5, 1.05]}
        xLabel="unified-agent vs. full-recompute decision agreement (note: cumulative-TTFT speedup)"
        refX={[{ x: 1, label: 'identical' }]}
        labelWidth={185}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, color: 'var(--ink-faint)', marginTop: 6 }}>
        10 domains × 10 instances = 300 decisions per model, except the two Gemma models (40
        instances, 120 decisions). Speedup scales with policy length × turns; the maximum observed
        here is {fmtX(Math.max(...ag.map((m: any) => m.speedup)), 1)}.
      </div>
    </div>
  )
}

export function Keystone() {
  return (
    <Section meta={META}>
      <P>
        Editing (§3) and composing (§2) both rest on the same mechanism (§7) — but are they really
        two operations on <em>one object</em>? The keystone test: transplant a skill whose body
        contains a mutable field, then <strong>edit the field inside the transplant</strong>. If
        the notes a transplant pastes in are the same notes an edit amends, every editing method
        should behave identically in a composed cache and a normally-recomputed one.
      </P>

      <Figure
        narrow
        label="The keystone."
        title="Editing inside a transplanted skill — points on the diagonal"
        caption={
          <>
            Each point is one editing method applied inside a transplanted skill (y) vs. inside a
            recomputed context (x). The editing mechanism reproduces verbatim — in-place weak,
            selective recovers with K, erratum strongest — and <b>composed ≈ recomputed for every
            method</b>. One substrate.
          </>
        }
      >
        <KeystoneScatter />
      </Figure>

      <H3>Both operations, live, across thirteen models</H3>
      <P>
        The paper embodies the claim in a single agent loop: a long policy is <em>composed</em>{' '}
        once and never re-prefilled; as the world changes across turns, mutable state is{' '}
        <em>edited</em> by appended errata; each turn prefills only the delta. Against a
        reprefill-every-turn baseline:
      </P>

      <Figure
        narrow
        label="The unified edit+compose agent."
        caption={
          <>
            Decision agreement with full recompute (bars, bootstrap CIs) and cumulative
            time-to-first-token speedup (annotations) across thirteen models spanning 0.6B–70B,
            dense, MoE, FP8 and 4-bit. Editing and composing operate together, losslessly and
            faster, on one cache.
          </>
        }
      >
        <UnifiedAgent />
      </Figure>

      <Aside>
        <b>Reading the agreement numbers.</b> Agreement is measured against full recompute on
        gated decisions that sit near the action boundary, so sub-percent logit differences can
        flip a discrete choice — the same boundary noise quantified in §10&rsquo;s 28-turn stress
        test, which shows it does <em>not</em> compound with trajectory length.
      </Aside>
    </Section>
  )
}
