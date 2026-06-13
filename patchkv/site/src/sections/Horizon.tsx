import { useState } from 'react'
import { Section, P, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { COLORS, Legend } from '../components/charts/core'
import { fmt } from '../lib/format'
import horizon from '../data/horizon.json'

const META = { id: 'horizon', num: '12', title: 'No compounding error over a long trajectory' }

const MODEL_COLORS = [COLORS.blue, COLORS.orange, COLORS.purple]

export function Horizon() {
  const runs = horizon.runs as any[]
  const [period, setPeriod] = useState(1)
  const sel = runs.filter((r) => r.period === period)

  return (
    <Section meta={META}>
      <P>
        A standing risk for any leave-stale scheme: small per-edit errors that <em>compound</em>{' '}
        over a long agent trajectory. The stress test: a gated clearance field toggles between a
        granting and a denying value {period === 1 ? 'every turn' : 'every six turns'} for 28
        turns; ONE evolving cache — static prefix reused forever, every change applied as an
        appended erratum, downstream never recomputed — is compared per-turn against a full
        reprefill of the byte-identical token sequence. The only difference is whether downstream
        KV was ever recomputed.
      </P>

      <Figure
        label="28 turns, one evolving cache."
        caption={
          <>
            Decision-logit cosine per turn stays flat at 0.99+ across three families — first-third
            vs last-third:{' '}
            {sel.map((r) => `${r.label} ${fmt(r.cos_first_third, 3)}→${fmt(r.cos_last_third, 3)}`).join(' · ')}.
            There is no drift with trajectory length. Discrete decision agreement is high but
            noisier (these gated decisions sit near the action boundary, where sub-percent logit
            differences can flip a discrete choice) — boundary noise, not compounding error.
          </>
        }
      >
        <Controls>
          <ControlGroup label="field toggles">
            <Seg options={[1, 6] as any} value={period as any} onChange={(v) => setPeriod(v as any)}
              labels={{ 1: 'every turn (p=1)', 6: 'every 6 turns (p=6)' } as any} />
          </ControlGroup>
        </Controls>
        <LineChart
          series={sel.map((r, i) => ({
            id: r.label, color: MODEL_COLORS[i % 3],
            points: r.per_turn.map((t: any) => ({ x: t.t, y: t.logit_cos })),
            marker: false,
          }))}
          xLabel="turn"
          yLabel="decision-logit cosine vs. full reprefill"
          yDomain={[0.9, 1.005]}
          height={280}
          refLinesY={[{ y: 1, label: 'identical' }]}
        />
        <Legend items={sel.map((r, i) => ({ label: `${r.label} (mean agree ${fmt(r.mean_agree, 2)})`, color: MODEL_COLORS[i % 3] }))} />
      </Figure>

      <Aside>
        <b>Why this matters.</b> The unified agent (§8) and the memory agent (§11) both ride on
        leave-stale-plus-erratum caches for entire sessions. This is the experiment that says they
        may: the erratum is not a patch that decays — the amended notes are as stable as fresh
        ones.
      </Aside>
    </Section>
  )
}
