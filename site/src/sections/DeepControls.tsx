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

const META = { id: 'controls', num: '4', title: 'Stress-testing the account: conclusion ≠ content' }

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
        <ControlGroup label="rule trigger (the ONE differing token)">
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
        <span className="dim">   ← field held byte-identical in both variants{'\n\n'}</span>
        <span className="hl-rule">
          {gateParts[0]}
          <span className="hl-diff" title="the only differing token">{v.trigger}</span>
          {gateParts.slice(1).join(v.trigger)}
        </span>
        {'\n\n'}
        <span className="dim">user: {dis.request}{'\n'}</span>
        correct action → <b>{v.conclusion}</b>
      </div>

      <div style={{ marginTop: 14 }}>
        <BarsH
          items={[
            { label: 'patch the differing rule token only', value: m.trigger_only.mean, lo: m.trigger_only.ci?.[0], hi: m.trigger_only.ci?.[1], color: COLORS.gray },
            { label: 'patch the downstream notes only', value: m.notes.mean, lo: m.notes.ci?.[0], hi: m.notes.ci?.[1], color: COLORS.orange },
          ]}
          domain={[-0.1, 1.15]}
          xLabel={`recovery of the flipped conclusion — ${m.label}, n=${m.n}`}
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
        xLabel="number of downstream positions transplanted (k)"
        yLabel="decision recovery"
        yDomain={[-0.06, 1.06]}
        highlightX={k}
        height={290}
      />
      <Legend items={[
        { label: 'k highest-effect positions', color: COLORS.orange },
        { label: 'k random downstream positions', color: COLORS.gray, dash: true },
      ]} />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        k = {k}: top-{k} recovers <b>{fmt(cur.top.mean, 2)}</b>; {k} random downstream positions recover{' '}
        <b>{cur.rand ? fmt(cur.rand.mean, 2) : '—'}</b>. A few specific aggregator tokens carry the
        conclusion — not a diffuse code.
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
          full-note injection: recovery <b>{fmt(m.full_recovery.mean, 2)}</b>, follows the written
          lie <b>{fmt(m.follows_rate, 2)}</b> of the time
        </span>
      </Controls>
      <LineChart
        series={[
          { id: 'rec', color: COLORS.purple, points: m.dose.map((d: any) => ({ x: d.k, y: d.recovery, lo: d.ci?.[0], hi: d.ci?.[1] })), band: true },
          { id: 'follow', color: COLORS.blue, dash: true, points: m.dose.filter((d: any) => d.follow_rate != null).map((d: any) => ({ x: d.k, y: d.follow_rate })) },
        ]}
        xLog
        xTicks={m.dose.map((d: any) => d.k)}
        xLabel="number of note tokens overwritten with the opposite conclusion"
        yLabel="recovery / follow rate"
        yDomain={[-0.06, 1.1]}
        height={280}
      />
      <Legend items={[
        { label: 'decision recovery toward the injected conclusion', color: COLORS.purple },
        { label: 'follow rate (decision obeys the false note)', color: COLORS.blue, dash: true },
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
      <text x={x0} y={16} className="tick-label">layer depth 0</text>
      <text x={x1} y={16} className="tick-label" textAnchor="end">1 (last layer)</text>
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
              write {fmt(r.write_depth, 2)}
            </text>
            <text x={x(r.commit_depth)} y={y + 21} textAnchor="middle" className="tick-label" style={{ fill: COLORS.blue, fontWeight: 700 }}>
              commit {fmt(r.commit_depth, 2)}
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
    { key: 'multihop', label: 'multi-hop reasoning' },
    { key: 'natural', label: 'free-form conversation' },
    { key: 'rag_lookup', label: 'near-verbatim RAG lookup' },
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
      yLabel="field-only recovery"
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
          <th>probe</th>
          {rep.map((r) => <th key={r.tag}>{r.label}</th>)}
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>field-only recovery (≈0 expected)</td>
          {rep.map((r) => <td key={r.tag}><b>{fmt(r.primary?.field_only?.mean ?? r.primary?.field_only, 3)}</b></td>)}
        </tr>
        <tr>
          <td>full-downstream recovery</td>
          {rep.map((r) => <td key={r.tag}>{fmt(r.primary?.full_downstream?.mean ?? r.primary?.full_downstream, 2)}</td>)}
        </tr>
        <tr>
          <td>dissociation: trigger-only / notes</td>
          {rep.map((r) => (
            <td key={r.tag}>
              {fmt(r.dissoc?.trigger_only?.mean ?? r.dissoc?.trigger_only, 2)} / <b>{fmt(r.dissoc?.notes?.mean ?? r.dissoc?.notes, 2)}</b>
            </td>
          ))}
        </tr>
        <tr>
          <td>specificity: top-k / random-k</td>
          {rep.map((r) => (
            <td key={r.tag}>
              <b>{fmt(r.specificity?.top_k?.mean ?? r.specificity?.top_k, 2)}</b> / {fmt(r.specificity?.random_k?.mean ?? r.specificity?.random_k, 2)}
            </td>
          ))}
        </tr>
        <tr>
          <td>false-note injection: recovery / follows</td>
          {rep.map((r) => (
            <td key={r.tag}>
              {fmt(r.injection?.recovery?.mean ?? r.injection?.recovery, 2)} / {fmt(r.injection?.follow_rate?.mean ?? r.injection?.follow_rate, 2)}
            </td>
          ))}
        </tr>
        <tr>
          <td>write depth → commit depth</td>
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
        A skeptic&rsquo;s null hypothesis survives §2: maybe the downstream tokens merely{' '}
        <em>re-encode the field&rsquo;s content</em>, and the decision re-reads that copy. Four
        controls close the gap between &ldquo;the conclusion is <em>decodable</em>{' '}
        downstream&rdquo; and &ldquo;the decision <em>uses</em> a memoized conclusion.&rdquo;
      </P>

      <H3>Control 1 — dissociating conclusion from content</H3>
      <P>
        Hold the field value <strong>byte-identical</strong> and flip a single rule token — a
        polarity <em>trigger</em> — so the conclusion inverts while the content does not. If the
        notes carried field content, patching them across this pair would do nothing (the field
        never changed). Toggle the trigger:
      </P>
      <Figure
        label="Dissociation."
        caption={
          <>
            The two prompts differ in exactly one token, yet transplanting the downstream notes
            alone carries the whole flipped conclusion (recovery ≈1.0) while patching the differing
            rule token itself carries none (≈0). The decision cannot be re-encoding field content —
            the content is constant. (A linear probe finds <em>both</em> conclusion and field
            identity decodable downstream; only the causal patch separates them.)
          </>
        }
      >
        <Dissociation />
      </Figure>

      <H3>Control 2 — a few specific tokens, not a diffuse code</H3>
      <Figure
        label="Specificity."
        caption={
          <>
            Transplanting the eight highest-effect downstream positions recovers 0.74–0.79 of the
            decision across models; eight random downstream positions recover ≤0.035.
          </>
        }
      >
        <Specificity />
      </Figure>

      <H3>Control 3 — the note is writable: inject a lie</H3>
      <P>
        The strongest test of &ldquo;the decision reads the notes&rdquo; is to <em>write</em> the
        notes. Take an internally consistent cache — field and rule agree — and overwrite just the
        note positions with KV carrying the <em>opposite</em> conclusion:
      </P>
      <Figure
        label="False-note injection."
        caption={
          <>
            The decision follows the written note <em>against the model&rsquo;s own live field</em>{' '}
            (recovery ≈1.0, follow rate 1.0), and a handful of note tokens suffice. The cache is
            not a passive transcript; it is the operative record of what the model believes it
            concluded.
          </>
        }
      >
        <Injection />
      </Figure>

      <H3>Control 4 — written before it is read</H3>
      <Figure
        narrow
        label="Timing."
        caption={
          <>
            Within the single prefill pass, the conclusion becomes linearly decodable on the
            downstream aggregator at depth 0.19–0.39 — roughly twelve layers <em>before</em> the
            decision token&rsquo;s logit-lens commit at depth 0.47–0.77. The note is written, then
            read. Dots from the released timing records (Gemma/Mistral) and the deep-mechanism
            study (Qwen/Llama). <PaperConst src={constants.timing_depths.source} />
          </>
        }
      >
        <Timing />
      </Figure>

      <H3>Does it survive off the template?</H3>
      <Figure
        label="Generalization, with an honest boundary."
        caption={
          <>
            Field-only recovery stays ≈0 for multi-hop reasoning and free-form conversational
            phrasing — the mechanism is not a template artifact. For near-verbatim attribute
            lookup it is bounded (0.25–0.63): there the &ldquo;note&rdquo; is partly a literal
            copy of the field, and refreshing the field does partial work. The paper reports this
            boundary rather than hiding it.
          </>
        }
      >
        <Generalization />
      </Figure>

      <H3>Not a model-family artifact</H3>
      <Figure
        narrow
        label="Four families."
        caption={
          <>
            The full probe battery replicated on two further architecture families with a
            tokenizer-robust readout (Gemma-2 keeps its attention/logit soft-capping intact).
            Every result holds: Qwen3, Llama-3.1, Gemma-2, Mistral.
          </>
        }
      >
        <Replication />
      </Figure>

      <Aside>
        <b>Where this leaves the account.</b> The conclusion is causally separable from the
        content (Control 1), concentrated on nameable positions (Control 2), writable (Control 3),
        and written at prefill before any read (Control 4). What remains is to find the components
        — §5 names the heads.
      </Aside>
    </Section>
  )
}
