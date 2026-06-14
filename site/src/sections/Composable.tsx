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

const META = { id: 'composable', num: '2', title: 'Reuse a skill instantly' }

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
        xLabel="length of the skill (tokens)"
        yLabel="time until the first word of the reply"
        yDomain={[Math.min(...m.points.map((p: any) => p.precomp_ms)) * 0.7, Math.max(...m.points.map((p: any) => p.full_ms)) * 1.4]}
        height={300}
      />
      <Legend items={[
        { label: 'read the skill fresh every time', color: COLORS.red },
        { label: 'paste the saved notes (re-stamp + slot in)', color: COLORS.blue },
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
            pasted notes match fresh reading: <b>{m.summary.agree[0]}/{m.summary.agree[1]} skills</b> (mean similarity {fmt(m.summary.cos, 3)})
          </span>
        )}
      </Controls>
      <table className="data-table">
        <thead>
          <tr><th>skill area</th><th>skill size (tokens)</th><th>read fresh</th><th>pasted notes</th><th>similarity</th><th>same decision</th></tr>
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
        read straight from the released run logs (<code>comp_div_*.log</code>); the skills span refund
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
      xLabel="how closely pasted notes match reading the skill fresh"
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
      xLabel="how often pasted notes and fresh reading pick the same tool (N=108 real tool calls)"
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
        xLabel="number of separately-saved skills pasted into one prompt"
        yLabel="how closely the result matches reading them fresh"
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
        Here is the idea. A reusable skill — a refund policy, a tool manual — can be read{' '}
        <strong>once</strong> and reused anywhere, instead of being re-read every time it shows up in
        a new prompt. As the model reads a prompt, it jots down a private notebook of notes about
        what it just read (engineers call this the &ldquo;KV cache&rdquo;). So we let the model read
        the skill once on its own, save that notebook, and later paste those saved notes into any new
        prompt. Each note carries a kind of &ldquo;position stamp&rdquo; that records where in the
        prompt it came from. To slot the notes into a new spot, we simply re-stamp them with their
        new positions — nothing else about the notes changes. Why this still works — the notes only
        summarize the skill&rsquo;s own text, which the model can still see — is the mechanism we
        unpack in §7.
      </P>

      <Figure
        label="Re-stamping the notes for a new spot."
        title="Re-stamping a note is just turning a dial"
        caption={
          <>
            The skill&rsquo;s notes were saved with position stamps for slots 0…L−1. To place them
            later in the prompt, each stamp is turned forward by the gap; the rest of each note is
            copied as-is. The note-saving machinery itself is prior work (Prompt Cache, CacheBlend,
            EPIC, CacheSlide) — this paper&rsquo;s contribution is explaining <em>when this keeps the
            model&rsquo;s behavior the same</em>, and showing on real decisions that it does.
          </>
        }
      >
        <RopeRotation />
      </Figure>

      <Figure
        label="Faster, and far faster as skills grow."
        caption={
          <>
            Reading the skill fresh gets disproportionately slower as the skill grows. Pasting the
            saved notes only takes a quick re-stamp, so the time grows in step with the skill&rsquo;s
            length instead. The longer the skill, the bigger the win: 13.9× faster at a 32,000-token
            skill on an 8-billion-parameter model.
          </>
        }
      >
        <TtftScaling />
      </Figure>

      <H3>But does the pasted skill still steer the answer?</H3>
      <P>
        Speed is the easy half. The real test is whether the skill still does its job: after the
        notes are pasted in, does the model make the same decision it would have made if it had read
        the skill fresh?
      </P>

      <Figure
        narrow
        label="Eight skill areas, one model at a time."
        caption={
          <>
            A saved-and-re-stamped skill produces the same decision as reading it fresh, area by
            area. The one lingering glitch in the whole study is at the very start of the pasted
            notes — the first words, which originally expected some text in front of them that is now
            gone. Re-reading just 1–2 of those opening words fixes it. §7&rsquo;s mechanism explains
            why it is only those opening words that need the touch-up.
          </>
        }
      >
        <DomainScorecard />
      </Figure>

      <Figure
        narrow
        label="Holds up across many models."
        caption={
          <>
            How closely pasting the saved notes matches reading the skill fresh, across model
            families and sizes, including compressed and mixture-of-experts versions.{' '}
            <PaperConst src={constants.transplant_cosine_bar.source} />
          </>
        }
      >
        <FidelityBar />
      </Figure>

      <Figure
        narrow
        label="Real tool use, measured on actual tool calls."
        caption={
          <>
            Pasting the notes keeps the model picking the right tools — measured on real tool calls,
            not a stand-in score — across 8 models. The one steady exception in the wider study is a
            model that reads through a moving window (Gemma); we diagnose and fix it in §11.
          </>
        }
      >
        <Agentic />
      </Figure>

      <Figure
        narrow
        label="Stack several skills together."
        caption={
          <>
            How closely the result matches reading everything fresh as 1 to 4 separately-saved skills
            are pasted into one prompt. The decision matches fresh reading in nearly every case; the
            rare misses are still a near-perfect match (similarity ≥ 0.996) and happen only when the
            decision was a coin-flip to begin with — the same sensitivity measured in §10, not damage
            from pasting.
          </>
        }
      >
        <MultiSkill />
      </Figure>

      <Aside>
        <b>Why it works, in one line.</b> The saved notes only sum up{' '}
        <em>the skill&rsquo;s own text</em>, which the model can still see. So moving them to a new
        spot changes nothing but the position stamp — and re-stamping is exact, with nothing lost.
      </Aside>
    </Section>
  )
}
