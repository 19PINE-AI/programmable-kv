import { useState } from 'react'
import { Section, P, H3, Aside, PaperConst } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH, BarsV } from '../components/charts/BarCI'
import { ChartSvg, COLORS, Legend } from '../components/charts/core'
import { fmt } from '../lib/format'
import controls from '../data/controls.json'
import prompts from '../data/prompts.json'
import constants from '../data/constants.json'

const META = { id: 'controls', num: '12', title: 'Under the hood: four careful checks' }

/* ------------------------------------------------------------------ */
/* (a) dissociation: one trigger token flips the conclusion           */
/* ------------------------------------------------------------------ */

function Dissociation() {
  const dis = prompts.dissociation as any
  const [variant, setVariant] = useState(0)
  const [tag, setTag] = useState('qwen3_8b')
  const xc = (controls.xcond as any[])
  const m = xc.find((x) => x.tag === tag)!
  const v = dis.variants[variant]

  const gateParts = v.gate.split(v.trigger)

  return (
    <div>
      <Controls>
        <ControlGroup label="the ONE word we change">
          <Seg options={[0, 1] as any} value={variant as any} onChange={(x) => setVariant(x as any)}
            labels={{ 0: dis.variants[0].trigger, 1: dis.variants[1].trigger } as any} accent="orange" />
        </ControlGroup>
        <ControlGroup label="model">
          <ModelPicker models={xc.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
      </Controls>

      <div className="prompt-box" style={{ maxHeight: 200 }}>
        <span className="dim">SESSION CONTEXT{'\n'}</span>
        <span className="hl-field">{dis.field_label}: {dis.field_value}</span>
        <span className="dim">   ← the fact stays exactly the same in both cases{'\n\n'}</span>
        <span className="hl-rule">
          {gateParts[0]}
          <span className="hl-diff" title="the only word that changes">{v.trigger}</span>
          {gateParts.slice(1).join(v.trigger)}
        </span>
        {'\n\n'}
        <span className="dim">user: {dis.request}{'\n'}</span>
        correct action → <b>{v.conclusion}</b>
      </div>

      <div style={{ marginTop: 14 }}>
        <BarsH
          items={[
            { label: 'copy over only the one word that changed', value: m.trigger_only.mean, lo: m.trigger_only.ci?.[0], hi: m.trigger_only.ci?.[1], color: COLORS.gray },
            { label: "copy over only the model's later notes", value: m.notes.mean, lo: m.notes.ci?.[0], hi: m.notes.ci?.[1], color: COLORS.orange },
          ]}
          domain={[-0.1, 1.15]}
          xLabel={`how much of the flipped answer comes back — ${m.label}, n=${m.n}`}
          refX={[{ x: 0, label: 'none' }, { x: 1, label: 'all' }]}
          labelWidth={230}
        />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* (b) specificity: top-k vs random-k                                 */
/* ------------------------------------------------------------------ */

function Specificity() {
  const sp = controls.specificity as any[]
  const [tag, setTag] = useState('qwen3_8b')
  const m = sp.find((x) => x.tag === tag)!
  const [k, setK] = useState(8)
  const ks = m.ks.map((x: any) => x.k)
  const cur = m.ks.find((x: any) => x.k === k) ?? m.ks[0]

  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={sp.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
        <ControlGroup label={`k = ${k} tokens`}>
          <input type="range" className="slider" min={0} max={ks.length - 1} step={1}
            value={ks.indexOf(k)} onChange={(e) => setK(ks[parseInt(e.target.value)])} style={{ width: 160 }} />
        </ControlGroup>
      </Controls>
      <LineChart
        series={[
          { id: 'top', color: COLORS.orange, points: m.ks.map((x: any) => ({ x: x.k, y: x.top.mean, lo: x.top.ci?.[0], hi: x.top.ci?.[1] })), band: true },
          { id: 'rand', color: COLORS.gray, dash: true, points: m.ks.filter((x: any) => x.rand).map((x: any) => ({ x: x.k, y: x.rand.mean, lo: x.rand.ci?.[0], hi: x.rand.ci?.[1] })) },
        ]}
        xLog
        xTicks={ks}
        xLabel="how many note spots we copy over (k)"
        yLabel="how much of the answer comes back"
        yDomain={[-0.06, 1.06]}
        highlightX={k}
        height={290}
      />
      <Legend items={[
        { label: 'the k spots that matter most', color: COLORS.orange },
        { label: 'k randomly chosen note spots', color: COLORS.gray, dash: true },
      ]} />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        k = {k}: the {k} most important spots bring back <b>{fmt(cur.top.mean, 2)}</b> of the answer;
        {' '}{k} random spots bring back{' '}
        <b>{cur.rand ? fmt(cur.rand.mean, 2) : '—'}</b>. A few specific spots hold the answer —
        it is not spread thinly everywhere.
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* (c) false-note injection                                           */
/* ------------------------------------------------------------------ */

function Injection() {
  const inj = controls.inject as any[]
  const [tag, setTag] = useState('qwen3_8b')
  const m = inj.find((x) => x.tag === tag)!
  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={inj.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
        <span style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)' }}>
          overwrite the whole note: answer comes back <b>{fmt(m.full_recovery.mean, 2)}</b>, and the
          model follows the false note <b>{fmt(m.follows_rate, 2)}</b> of the time
        </span>
      </Controls>
      <LineChart
        series={[
          { id: 'rec', color: COLORS.purple, points: m.dose.map((d: any) => ({ x: d.k, y: d.recovery, lo: d.ci?.[0], hi: d.ci?.[1] })), band: true },
          { id: 'follow', color: COLORS.blue, dash: true, points: m.dose.filter((d: any) => d.follow_rate != null).map((d: any) => ({ x: d.k, y: d.follow_rate })) },
        ]}
        xLog
        xTicks={m.dose.map((d: any) => d.k)}
        xLabel="how many note spots we overwrite with the opposite answer"
        yLabel="answer comes back / model follows the note"
        yDomain={[-0.06, 1.1]}
        height={280}
      />
      <Legend items={[
        { label: 'the answer shifts toward the note we wrote in', color: COLORS.purple },
        { label: 'how often the answer obeys the false note', color: COLORS.blue, dash: true },
      ]} />
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* (d) timing: written before it is read                              */
/* ------------------------------------------------------------------ */

function Timing() {
  const td = constants.timing_depths
  const rep = controls.replicate as any[]
  const rows = [
    ...td.rows.map((r: any) => ({ ...r, src: 'jsonish' })),
    ...rep.map((r) => ({
      label: r.label, nlayers: r.timing.nlayers,
      write_layer: r.timing.write_layer, write_depth: r.timing.write_depth,
      commit_layer: r.timing.commit_layer, commit_depth: r.timing.commit_depth,
    })),
  ]
  const W = 660
  const rowH = 46
  const H = rows.length * rowH + 56
  const x0 = 150
  const x1 = W - 30
  const x = (d: number) => x0 + d * (x1 - x0)

  return (
    <ChartSvg width={W} height={H}>
      <text x={x0} y={16} className="tick-label">start of the model</text>
      <text x={x1} y={16} className="tick-label" textAnchor="end">end of the model</text>
      {rows.map((r, i) => {
        const y = 36 + i * rowH
        return (
          <g key={r.label}>
            <text x={x0 - 10} y={y + 5} textAnchor="end" className="tick-label" style={{ fontSize: 11.5 }}>{r.label}</text>
            <line x1={x0} x2={x1} y1={y} y2={y} stroke="var(--rule-strong)" strokeWidth={1.5} />
            {/* gap band */}
            <rect x={x(r.write_depth)} y={y - 5} width={x(r.commit_depth) - x(r.write_depth)} height={10} fill={COLORS.orangeSoft} opacity={0.55} rx={5} />
            <circle cx={x(r.write_depth)} cy={y} r={6.5} fill={COLORS.orange} stroke="#fff" strokeWidth={1.5}>
              <title>{`write: layer ${r.write_layer}/${r.nlayers} (depth ${r.write_depth})`}</title>
            </circle>
            <circle cx={x(r.commit_depth)} cy={y} r={6.5} fill={COLORS.blue} stroke="#fff" strokeWidth={1.5}>
              <title>{`commit: layer ${r.commit_layer} (depth ${r.commit_depth})`}</title>
            </circle>
            <text x={x(r.write_depth)} y={y - 11} textAnchor="middle" className="tick-label" style={{ fill: COLORS.orange, fontWeight: 700 }}>
              note written {fmt(r.write_depth, 2)}
            </text>
            <text x={x(r.commit_depth)} y={y + 21} textAnchor="middle" className="tick-label" style={{ fill: COLORS.blue, fontWeight: 700 }}>
              answer decided {fmt(r.commit_depth, 2)}
            </text>
          </g>
        )
      })}
    </ChartSvg>
  )
}

/* ------------------------------------------------------------------ */
/* (e) off-template generalization                                    */
/* ------------------------------------------------------------------ */

function Generalization() {
  const gen = controls.general as any[]
  const fams: { key: string; label: string }[] = [
    { key: 'multihop', label: 'multi-step reasoning' },
    { key: 'natural', label: 'free-form conversation' },
    { key: 'rag_lookup', label: 'near word-for-word lookup' },
  ]
  return (
    <BarsV
      groups={fams.map((f) => ({
        label: f.label,
        values: gen.map((g) => {
          const fam = g.families[f.key]
          return fam ? { v: fam.field_only.mean, lo: fam.field_only.ci?.[0], hi: fam.field_only.ci?.[1] } : null
        }),
      }))}
      seriesLabels={gen.map((g) => g.label)}
      colors={[COLORS.orange, COLORS.blue, COLORS.purple]}
      yDomain={[-0.1, 1.0]}
      yLabel="answer recovered by refreshing the fact alone"
      height={280}
    />
  )
}

/* ------------------------------------------------------------------ */
/* (f) four-family replication strip                                  */
/* ------------------------------------------------------------------ */

function Replication() {
  const rep = controls.replicate as any[]
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>check</th>
          {rep.map((r) => <th key={r.tag}>{r.label}</th>)}
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>refreshing the fact alone (should be ≈0)</td>
          {rep.map((r) => <td key={r.tag}><b>{fmt(r.primary?.field_only?.mean ?? r.primary?.field_only, 3)}</b></td>)}
        </tr>
        <tr>
          <td>copying over all the later notes</td>
          {rep.map((r) => <td key={r.tag}>{fmt(r.primary?.full_downstream?.mean ?? r.primary?.full_downstream, 2)}</td>)}
        </tr>
        <tr>
          <td>check (i): one word changed / the notes</td>
          {rep.map((r) => (
            <td key={r.tag}>
              {fmt(r.dissoc?.trigger_only?.mean ?? r.dissoc?.trigger_only, 2)} / <b>{fmt(r.dissoc?.notes?.mean ?? r.dissoc?.notes, 2)}</b>
            </td>
          ))}
        </tr>
        <tr>
          <td>check (iii): top spots / random spots</td>
          {rep.map((r) => (
            <td key={r.tag}>
              <b>{fmt(r.specificity?.top_k?.mean ?? r.specificity?.top_k, 2)}</b> / {fmt(r.specificity?.random_k?.mean ?? r.specificity?.random_k, 2)}
            </td>
          ))}
        </tr>
        <tr>
          <td>check (iv): answer back / follows false note</td>
          {rep.map((r) => (
            <td key={r.tag}>
              {fmt(r.injection?.recovery?.mean ?? r.injection?.recovery, 2)} / {fmt(r.injection?.follow_rate?.mean ?? r.injection?.follow_rate, 2)}
            </td>
          ))}
        </tr>
        <tr>
          <td>check (ii): note written → answer decided</td>
          {rep.map((r) => (
            <td key={r.tag}>{fmt(r.timing.write_depth, 2)} → {fmt(r.timing.commit_depth, 2)}</td>
          ))}
        </tr>
      </tbody>
    </table>
  )
}

export function DeepControls() {
  return (
    <Section meta={META}>
      <P>
        This is an optional deep-dive. Earlier we saw the model jot down a worked-out answer as it
        reads &mdash; in what we&rsquo;ll call its &ldquo;notebook of notes&rdquo; (the running
        scratchpad it keeps while processing a prompt). But there&rsquo;s a fair worry: maybe those
        notes are just a <em>copy of the original fact</em>, and the model simply re-reads the copy.
        You can often <em>read</em> the answer off the notes &mdash; but that alone doesn&rsquo;t
        prove the model actually <em>uses</em> them. So we tested use directly, by swapping notes in
        and out and watching what the model does. Four careful checks tell the difference.
      </P>

      <H3>Check (i) — it&rsquo;s the conclusion, not a copy of the fact</H3>
      <P>
        Keep the underlying fact <strong>exactly the same</strong>, word for word, and change just
        one word in the rule &mdash; enough to flip the right answer. If the notes only held a copy
        of the fact, copying them between these two cases should change nothing, since the fact
        never moved. Flip the word and watch:
      </P>
      <Figure
        label="It&rsquo;s the conclusion, not a copy of the fact."
        caption={
          <>
            The two prompts differ by exactly one word. Yet copying over the model&rsquo;s later
            notes alone brings back the whole flipped answer (about 1.0), while copying that one
            changed word brings back almost none (about 0). The notes can&rsquo;t just be a copy of
            the fact &mdash; the fact never changed. (You <em>can</em> read both the answer and the
            fact off the notes; only this swap test shows which one the model actually relies on.)
          </>
        }
      >
        <Dissociation />
      </Figure>

      <H3>Check (iii) — a few specific spots carry it</H3>
      <Figure
        label="A few specific spots carry it."
        caption={
          <>
            Copying over just the eight most important note spots brings back 0.74&ndash;0.79 of the
            answer across models; eight randomly chosen spots bring back 0.035 or less. The answer
            lives in a few specific places, not smeared across the whole notebook.
          </>
        }
      >
        <Specificity />
      </Figure>

      <H3>Check (iv) — the model follows the note even when the note is wrong</H3>
      <P>
        The strongest way to show the model reads its notes is to <em>edit</em> them. Start from a
        case where everything agrees &mdash; the fact and the rule point to the same answer &mdash;
        and then overwrite just the note spots so they say the <em>opposite</em>:
      </P>
      <Figure
        label="The model follows the note even when the note is wrong."
        caption={
          <>
            The model goes with the note we wrote in, <em>even though its own fact still says the
            other thing</em> (answer comes back about 1.0, followed every time), and only a handful
            of note spots are needed. The notebook isn&rsquo;t a passive transcript &mdash; it&rsquo;s
            the working record of what the model thinks it concluded.
          </>
        }
      >
        <Injection />
      </Figure>

      <H3>Check (ii) — the note is written before it&rsquo;s read</H3>
      <Figure
        narrow
        label="The note is written before it&rsquo;s read."
        caption={
          <>
            As the model reads the prompt in a single pass, the answer shows up in its notes early
            on (about a fifth to two-fifths of the way through) &mdash; roughly twelve steps
            <em> before</em> the model commits to its final answer (about halfway to three-quarters
            through). The note is written first, then read. Dots come from two sets of released
            measurements (Gemma/Mistral) and the deeper study (Qwen/Llama).{' '}
            <PaperConst src={constants.timing_depths.source} />
          </>
        }
      >
        <Timing />
      </Figure>

      <H3>Does it hold up away from our test format?</H3>
      <Figure
        label="It holds up, with one honest exception."
        caption={
          <>
            Just refreshing the original fact brings back almost nothing (about 0) for multi-step
            reasoning and free-form conversation &mdash; so this isn&rsquo;t a quirk of our test
            format. There&rsquo;s one honest exception: when the task is looking up an attribute
            almost word for word, refreshing the fact does some of the work (0.25&ndash;0.63),
            because there the &ldquo;note&rdquo; really is partly just a copy of the fact. We report
            this exception rather than hide it.
          </>
        }
      >
        <Generalization />
      </Figure>

      <H3>Not specific to one kind of model</H3>
      <Figure
        narrow
        label="Four model families."
        caption={
          <>
            We ran the full set of checks on two more kinds of model, using a measurement that
            works across their different ways of splitting text into pieces (and leaving Gemma-2&rsquo;s
            internals untouched). Every result holds across all four: Qwen3, Llama-3.1, Gemma-2,
            and Mistral.
          </>
        }
      >
        <Replication />
      </Figure>

      <Aside>
        <b>Where this leaves us.</b> The note holds a worked-out answer, not just a copy of the
        fact (check i); it&rsquo;s written before it&rsquo;s read (check ii); a few specific spots
        carry it (check iii); and the model follows the note even when the note is wrong (check iv).
        What&rsquo;s left is to find the exact parts of the model that do this &mdash; §13 names them.
      </Aside>
    </Section>
  )
}
