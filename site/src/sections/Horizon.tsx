import { useState } from 'react'
import { Section, P, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { COLORS, Legend } from '../components/charts/core'
import { fmt } from '../lib/format'
import horizon from '../data/horizon.json'

const META = { id: 'horizon', num: '10', title: 'Do the errors pile up? No.' }

const MODEL_COLORS = [COLORS.blue, COLORS.orange, COLORS.purple]

export function Horizon() {
  const runs = horizon.runs as any[]
  const [period, setPeriod] = useState(1)
  const sel = runs.filter((r) => r.period === period)

  return (
    <Section meta={META}>
      <P>
        Here is the natural worry. The model keeps a notebook of notes about the conversation (its
        "KV cache"). Instead of rewriting that notebook from scratch every time something changes,
        our method just jots a small correction at the end and moves on. Over a long, many-turn
        chat, do those little corrections slowly add up into a mess? To find out, we flip one fact
        back and forth — a permission that switches between "allowed" and "denied"{' '}
        {period === 1 ? 'on every turn' : 'every six turns'} for 28 turns in a row. We run it two
        ways and compare them turn by turn: one notebook that we only ever amend, and a fresh
        notebook rewritten from scratch each turn. The only difference between them is whether the
        notes were redone.
      </P>

      <Figure
        label="28 turns, one amended notebook."
        caption={
          <>
            The line below shows how closely the amended notebook matches a fresh rewrite each
            turn — 1.0 means a perfect match. It stays flat above 0.99 for all three model
            families, with the start of the run looking just like the end —{' '}
            {sel.map((r) => `${r.label} ${fmt(r.cos_first_third, 3)}→${fmt(r.cos_last_third, 3)}`).join(' · ')}.
            In short, the answers do not drift as the conversation grows longer. How often the two
            versions land on the exact same yes/no choice is also high, just a touch noisier: these
            permission checks sit right on the line between yes and no, so a tiny difference can
            tip the call either way. That is just borderline jitter, not errors piling up.
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
          yLabel="match vs. fresh rewrite (1.0 = identical)"
          yDomain={[0.9, 1.005]}
          height={280}
          refLinesY={[{ y: 1, label: 'identical' }]}
        />
        <Legend items={sel.map((r, i) => ({ label: `${r.label} (same choice ${fmt(r.mean_agree, 2)})`, color: MODEL_COLORS[i % 3] }))} />
      </Figure>

      <Aside>
        <b>Why this matters.</b> The unified agent (§9) and the memory agent (§4) both run for a
        whole session on a notebook that we only ever amend, never redo. This is the test that says
        they can: a small correction does not slowly wear off, and the amended notes stay just as
        reliable as a fresh rewrite.
      </Aside>
    </Section>
  )
}
