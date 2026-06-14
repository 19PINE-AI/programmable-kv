import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { DiagonalScatter } from '../components/charts/DiagonalScatter'
import { BarsH } from '../components/charts/BarCI'
import { COLORS, Legend } from '../components/charts/core'
import { fmt, fmtX } from '../lib/format'
import keystone from '../data/keystone.json'

const META = { id: 'keystone', num: '9', title: 'Editing and reuse are the same trick' }

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
        xLabel="how well the edit took when the notes were made fresh"
        yLabel="how well the edit took inside reused notes"
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
        circles = Llama-3.1-8B (n=8) · squares = Gemma-2-9B; values show how much of the edit landed, and can edge past 1
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
        xLabel="how often the reuse-and-edit agent makes the same call as redoing it all (annotations: speedup)"
        refX={[{ x: 1, label: 'identical' }]}
        labelWidth={185}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, color: 'var(--ink-faint)', marginTop: 6 }}>
        10 domains × 10 instances = 300 decisions per model, except the two Gemma models (40
        instances, 120 decisions). The longer the instructions and the more turns, the bigger the
        speedup; the most we saw here is {fmtX(Math.max(...ag.map((m: any) => m.speedup)), 1)}.
      </div>
    </div>
  )
}

export function Keystone() {
  return (
    <Section meta={META}>
      <P>
        As the model reads, it builds up a private notebook of notes about what it has seen. Part I
        showed we can paste a ready-made skill straight into that notebook and reuse it. Part II
        showed we can edit a fact already written in the notebook. Here is the payoff that ties them
        together: both abilities act on the <em>same notes</em>. To prove it, we paste in a skill
        that contains a fact, then <strong>change that fact right inside the pasted-in skill</strong>.
        If reuse and editing really touch one notebook, then editing a pasted-in fact should work
        exactly as well as editing a fact the model wrote itself.
      </P>

      <Figure
        narrow
        label="The payoff."
        title="Editing a fact inside a pasted-in skill — every point lands on the line"
        caption={
          <>
            Each point compares one way of editing a fact inside a pasted-in skill (up the side)
            against editing that same fact the model wrote itself (along the bottom). The editing
            methods rank the same in both cases, and <b>pasted-in lines up with from-scratch every
            time</b>. Reuse and editing really are touching the same notes.
          </>
        }
      >
        <KeystoneScatter />
      </Figure>

      <H3>Both abilities at once, live, across thirteen models</H3>
      <P>
        We put both abilities to work in one running agent. It reads a long set of instructions{' '}
        <em>once</em> and reuses those notes for the rest of the task. As the situation changes from
        turn to turn, it <em>edits</em> the relevant facts in place instead of re-reading everything.
        We compare it against the slow way — re-reading the whole thing every single turn:
      </P>

      <Figure
        narrow
        label="One agent that reuses and edits."
        caption={
          <>
            How often the agent reaches the same decision as the slow re-read-everything way (bars,
            with confidence ranges), plus how much faster it responds (annotations), across thirteen
            models from 0.6B to 70B of all kinds. Reusing and editing work together, just as
            correctly and a lot faster, on one notebook.
          </>
        }
      >
        <UnifiedAgent />
      </Figure>

      <Aside>
        <b>Why agreement isn&rsquo;t a perfect 100%.</b> We score the agent on close-call decisions,
        right at the line where a yes could just as easily be a no. On those, a tiny rounding-sized
        difference can tip the choice the other way. Section 10&rsquo;s 28-turn stress test measures
        exactly this wobble and shows it does <em>not</em> pile up as the task runs longer.
      </Aside>
    </Section>
  )
}
