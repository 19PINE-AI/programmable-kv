import { useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH } from '../components/charts/BarCI'
import { ChartSvg, COLORS, Legend } from '../components/charts/core'
import { fmt } from '../lib/format'
import circuit from '../data/circuit.json'

const META = { id: 'circuit', num: '13', title: 'Under the hood: the wiring' }

function layerOf(head: string) {
  return parseInt(head.split('.')[0], 10)
}

/** Scatter of named heads: x = layer, y = single-head recovery; orange write / blue read. */
function HeadMap({ m }: { m: any }) {
  const W = 660
  const H = 300
  const pad = { l: 54, r: 16, t: 18, b: 44 }
  const heads = [...m.write_heads.map((h: any) => ({ ...h, kind: 'write' })), ...m.read_heads.map((h: any) => ({ ...h, kind: 'read' }))]
  const maxL = Math.max(...heads.map((h) => layerOf(h.head))) + 2
  const maxR = Math.max(...heads.map((h) => h.rec ?? 0), 0.5) * 1.15
  const x = (l: number) => pad.l + (l / maxL) * (W - pad.l - pad.r)
  const y = (r: number) => H - pad.b - (Math.max(r, 0) / maxR) * (H - pad.t - pad.b)
  const topRead = m.read_heads[0]
  const [hover, setHover] = useState<any | null>(null)

  return (
    <div style={{ position: 'relative' }}>
      <ChartSvg width={W} height={H}>
        <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} stroke={COLORS.grayLight} />
        <line x1={pad.l} x2={pad.l} y1={pad.t} y2={H - pad.b} stroke={COLORS.grayLight} />
        {[0, 0.25, 0.5].filter((v) => v < maxR).map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="#eceadf" />
            <text className="tick-label" x={pad.l - 6} y={y(v)} dy="0.32em" textAnchor="end">{v}</text>
          </g>
        ))}
        {[0, 8, 16, 24, 32, 40].filter((l) => l <= maxL).map((l) => (
          <text key={l} className="tick-label" x={x(l)} y={H - pad.b + 14} textAnchor="middle">{l}</text>
        ))}
        <text className="axis-label" x={(pad.l + W) / 2} y={H - 8} textAnchor="middle">layer</text>
        <text className="axis-label" transform={`translate(13,${(H - pad.b + pad.t) / 2}) rotate(-90)`} textAnchor="middle">
          single-head ablation recovery
        </text>

        {heads.map((h, i) => (
          <circle
            key={i}
            cx={x(layerOf(h.head))}
            cy={y(h.rec ?? 0)}
            r={h.kind === 'read' ? 6 : 5}
            fill={h.kind === 'read' ? COLORS.blue : COLORS.orange}
            opacity={0.85}
            stroke="#fff"
            strokeWidth={1.2}
            onMouseEnter={() => setHover(h)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {topRead && (
          <text x={x(layerOf(topRead.head)) + 9} y={y(topRead.rec) + 4} className="tick-label" style={{ fill: COLORS.blue, fontWeight: 700 }}>
            head {topRead.head}
          </text>
        )}
      </ChartSvg>
      {hover && (
        <div style={{
          position: 'absolute', top: 6, right: 8, background: 'var(--ink)', color: '#fff',
          fontFamily: 'var(--sans)', fontSize: 11.5, padding: '6px 10px', borderRadius: 5,
        }}>
          <b>{hover.kind} head {hover.head}</b> · recovery {fmt(hover.rec, 3)}
          {hover.attn != null && <> · attn→agg {fmt(hover.attn, 3)}</>}
        </div>
      )}
      <Legend items={[
        { label: 'look-back channels that READ the note when answering', color: COLORS.blue },
        { label: 'channels that help WRITE the note while reading the prompt', color: COLORS.orange },
      ]} />
    </div>
  )
}

function CumK({ m }: { m: any }) {
  return (
    <div>
      <LineChart
        series={[
          { id: 'read', color: COLORS.blue, points: m.read_cumk.map((p: any) => ({ x: p.k, y: p.mean, lo: p.ci?.[0], hi: p.ci?.[1] })), band: true },
          { id: 'write', color: COLORS.orange, points: m.write_cumk.map((p: any) => ({ x: p.k, y: p.mean, lo: p.ci?.[0], hi: p.ci?.[1] })), band: true },
        ]}
        xLabel="number of look-back channels turned on together"
        yLabel="how much of the decision comes back"
        yDomain={[-0.05, 1.0]}
        xTicks={[1, 2, 3, 5, 8, 12]}
        height={280}
        refLinesY={[
          ...(m.read_ctrl ? [{ y: m.read_ctrl.mean, label: `random heads ${fmt(m.read_ctrl.mean, 3)}`, color: COLORS.gray }] : []),
        ]}
      />
      <Legend items={[
        { label: 'reading channels — a few of them do almost all the work', color: COLORS.blue },
        { label: 'writing channels — no single one matters much (the writing is spread out)', color: COLORS.orange },
        { label: 'random channels (a sanity check) ≈ 0', color: COLORS.gray, dash: true },
      ]} />
    </div>
  )
}

function AttnMlp({ m }: { m: any }) {
  const c = m.components
  if (!c?.attn_per_layer) return null
  const n = c.attn_per_layer.length
  const W = 660
  const H = 220
  const pad = { l: 54, r: 16, t: 14, b: 40 }
  const total = c.attn_per_layer.map((a: number, i: number) => a + (c.mlp_per_layer?.[i] ?? 0))
  const maxT = Math.max(...total, 1e-9)
  const bw = (W - pad.l - pad.r) / n
  const y = (v: number) => H - pad.b - (v / maxT) * (H - pad.t - pad.b)

  return (
    <div>
      <ChartSvg width={W} height={H}>
        <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} stroke={COLORS.grayLight} />
        {c.attn_per_layer.map((a: number, i: number) => {
          const ml = c.mlp_per_layer?.[i] ?? 0
          return (
            <g key={i}>
              <rect x={pad.l + i * bw + 1} y={y(Math.max(a, 0))} width={bw - 2} height={Math.max(0, H - pad.b - y(Math.max(a, 0)))} fill={COLORS.orange} opacity={0.9}>
                <title>{`layer ${i}: attention write ${fmt(a, 3)}`}</title>
              </rect>
              <rect x={pad.l + i * bw + 1} y={y(Math.max(a, 0) + Math.max(ml, 0))} width={bw - 2} height={Math.max(0, y(Math.max(a, 0)) - y(Math.max(a, 0) + Math.max(ml, 0)))} fill={COLORS.purple} opacity={0.75}>
                <title>{`layer ${i}: MLP write ${fmt(ml, 3)} ${ml < 0 ? '(negative, clamped in display)' : ''}`}</title>
              </rect>
              {i % 2 === 0 && (
                <text className="tick-label" x={pad.l + i * bw + bw / 2} y={H - pad.b + 13} textAnchor="middle" style={{ fontSize: 9 }}>{i}</text>
              )}
            </g>
          )
        })}
        <text className="axis-label" x={(pad.l + W) / 2} y={H - 8} textAnchor="middle">
          processing stage (up to the readout stage L{c.readout_layer})
        </text>
        <text className="axis-label" transform={`translate(13,${(H - pad.b + pad.t) / 2}) rotate(-90)`} textAnchor="middle">
          how much it adds to the note
        </text>
      </ChartSvg>
      <Legend items={[
        { label: `look-back channels write most of the note — share ${fmt(c.attn_share?.mean, 2)}`, color: COLORS.orange },
        { label: `per-position processing — share ${fmt(c.mlp_share?.mean, 2)}`, color: COLORS.purple },
      ]} />
    </div>
  )
}

function Direction({ m }: { m: any }) {
  if (!m.direction) return null
  const layers: number[] = m.direction.layers
  const pl = m.direction.per_layer
  const series = [
    { id: 'full', label: 'copy the whole note across', color: COLORS.gray, key: 'full', dash: true },
    { id: 'along', label: 'copy only the one "which conclusion" setting', color: COLORS.blue, key: 'along' },
    { id: 'random', label: 'copy a random setting (a sanity check)', color: COLORS.red, key: 'random', dash: true },
  ]
  return (
    <div>
      <LineChart
        series={series.map((s) => ({
          id: s.id, color: s.color, dash: s.dash,
          points: layers.filter((L) => pl[String(L)]).map((L) => {
            const v = pl[String(L)].dm[s.key]
            return { x: L, y: v.mean, lo: v.ci?.[0], hi: v.ci?.[1] }
          }),
          band: s.id === 'along',
        }))}
        xLabel="processing stage"
        yLabel="how much of the decision carried over"
        xTicks={layers}
        height={270}
        yDomain={[-0.06, Math.max(...layers.map((L) => pl[String(L)]?.dm.full.mean ?? 0)) * 1.2]}
      />
      <Legend items={series.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />
    </div>
  )
}

function Scrub({ m }: { m: any }) {
  if (!m.scrub) return null
  const s = m.scrub
  return (
    <BarsH
      items={[
        { label: 'swap in everything ELSE from a run that reached the same conclusion (should not move)', value: s.drift?.drift_rest_same?.mean, lo: s.drift?.drift_rest_same?.ci?.[0], hi: s.drift?.drift_rest_same?.ci?.[1], color: COLORS.gray },
        { label: 'swap in the NOTE from a run that reached the same conclusion (should not move)', value: s.drift?.drift_note_same?.mean, lo: s.drift?.drift_note_same?.ci?.[0], hi: s.drift?.drift_note_same?.ci?.[1], color: COLORS.gray },
        { label: `swap in the NOTE from a run that reached the OPPOSITE conclusion (k=${s.k_note})`, value: s.interchange?.rec_note_opp?.mean, lo: s.interchange?.rec_note_opp?.ci?.[0], hi: s.interchange?.rec_note_opp?.ci?.[1], color: COLORS.orange },
        { label: 'swap in everything ELSE from a run that reached the OPPOSITE conclusion', value: s.interchange?.rec_rest_opp?.mean, lo: s.interchange?.rec_rest_opp?.ci?.[0], hi: s.interchange?.rec_rest_opp?.ci?.[1], color: COLORS.blue },
      ]}
      domain={[0, 1.0]}
      xLabel="how far the answer moved toward the swapped-in conclusion"
      labelWidth={300}
    />
  )
}

function Sae() {
  const sae = (circuit as any).sae
  const L = sae?.L14
  if (!L) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
      <div>
        <div className="fig-sub" style={{ marginBottom: 6 }}>can it spot the conclusion? — accuracy of each learned detector</div>
        <BarsH
          items={L.top_features_auc.slice(0, 6).map(([feat, auc]: [number, number]) => ({
            label: `detector #${feat}`, value: auc, color: auc >= 0.99 ? COLORS.green : COLORS.blueSoft as any,
          }))}
          domain={[0, 1.05]}
          xLabel="accuracy at spotting the conclusion (1.0 = perfect)"
          labelWidth={110}
          width={420}
        />
      </div>
      <div>
        <div className="fig-sub" style={{ marginBottom: 6 }}>does it drive the answer? — using only the top detectors</div>
        <BarsH
          items={[
            ...L.sufficiency_byK.map((x: any) => ({ label: `top ${x.K} detectors`, value: x.mean, color: COLORS.orange })),
            { label: 'random detectors (a sanity check)', value: L.control_byK?.[L.control_byK.length - 1]?.mean ?? 0, color: COLORS.gray },
          ]}
          domain={[-0.05, 1.05]}
          xLabel="how much of the decision comes back"
          labelWidth={150}
          width={420}
        />
      </div>
    </div>
  )
}

export function Circuit() {
  const models = (circuit as any).models
  const [tag, setTag] = useState('llama31_8b')
  const m = models.find((x: any) => x.tag === tag)!

  return (
    <Section meta={META}>
      <P>
        This is an optional deep-dive. Here we pop the hood and look at the actual machinery,
        so it gets a bit more detailed — but we'll keep it in plain terms. Quick reminder: the
        model keeps a kind of running notebook of notes as it reads (researchers call it the
        &ldquo;KV cache&rdquo;). Earlier sections showed that editing one note can change the
        model's answer. Now we trace which parts of the model put that note there, and which
        parts read it back.
      </P>
      <P>
        One clear pattern shows up across all four model families. We call it{' '}
        <strong>&ldquo;distributed write, concentrated read.&rdquo;</strong> Many parts of the
        model help write the conclusion into the notes — the work is spread out and redundant, so
        there's no single piece you could remove to stop it. But only a small, identifiable handful
        of &ldquo;look-back channels&rdquo; read that note back out when the model answers. (A
        look-back channel is a specialized part of the model that, at each step, decides which
        earlier notes to glance back at; researchers call these &ldquo;attention heads.&rdquo;)
      </P>

      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={models.map((x: any) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
      </Controls>

      <Figure
        label="Experiment 1 — the look-back channels."
        title="A few channels read the note; many channels help write it"
        caption={
          <>
            Each dot is one look-back channel (hover for its id and score). The blue ones read
            the note when the model answers; there are only a few, and each one matters a lot — the
            single strongest channel ({m.read_heads[0]?.head}) on its own brings back{' '}
            {fmt(m.read_heads[0]?.rec, 2)} of the decision on {m.label}. The orange ones, which
            help write the note as the model reads the prompt, are many and each does little on its
            own. Picking channels at random brings back almost nothing ({fmt(m.read_ctrl?.mean, 3)}).
          </>
        }
      >
        <HeadMap m={m} />
      </Figure>

      <Figure
        label="Adding channels one at a time."
        caption={
          <>
            Turn on the reading channels a few at a time and the decision snaps back fast (the top
            twelve together: {fmt(m.read_cumk?.[m.read_cumk.length - 1]?.mean, 2)} on {m.label}).
            Do the same with the writing channels and it never climbs much — the writing really is
            spread out, so there's no one &ldquo;note-writing channel&rdquo; you could disable. This
            lopsidedness is exactly why editing the notes works better than trying to edit the
            model's wiring.
          </>
        }
      >
        <CumK m={m} />
      </Figure>

      <Figure
        label="Experiment 2 — look-back vs. per-position processing."
        title="The note is mostly copied in, not computed on the spot"
        caption={
          <>
            What actually builds the note? The model has two kinds of machinery: the look-back
            channels (which pull in information from earlier in the text) and a per-position
            processing layer (which crunches each spot on its own; researchers call it the
            &ldquo;MLP&rdquo;). Stage by stage, the look-back channels do most of the writing
            ({fmt(m.components?.attn_share?.mean, 2)} of it on {m.label}). In other words, the
            model is mostly copying a conclusion it already worked out elsewhere into the note,
            rather than building it fresh at this spot.
          </>
        }
      >
        <AttnMlp m={m} />
      </Figure>

      <Figure
        label="Experiment 3 — the one setting that carries the conclusion."
        caption={
          <>
            Inside the note there's essentially a single dial that records which conclusion the
            model reached. Copy just that one dial from another run and the decision follows it —
            almost as well as copying the whole note, and far better than copying some random dial.
            The effect peaks mid-way through the model, exactly where our earlier timing analysis
            said the note gets written. So the note has simple structure — but, as the next panels
            show, no single ingredient is the whole story.
          </>
        }
      >
        <Direction m={m} />
      </Figure>

      <Figure
        narrow
        label="Experiment 4 — the swap test."
        title="The note alone decides the answer"
        caption={
          <>
            The clinching test is to swap pieces between runs and watch what moves the answer
            (researchers call this &ldquo;causal scrubbing&rdquo;). Swap in a note from a run that
            reached the same conclusion and nothing changes (about{' '}
            {fmt(m.scrub?.drift?.drift_note_same?.mean, 2)}). Swap in a note from a run that reached
            the <em>opposite</em> conclusion and the answer flips to match it (
            {fmt(m.scrub?.interchange?.rec_note_opp?.mean, 2)}). But swap in everything <em>except</em>{' '}
            the note and the answer mostly stays put (
            {fmt(m.scrub?.interchange?.rec_rest_opp?.mean, 2)}). So it's the note itself, and not the
            surrounding context, that drives the decision.
          </>
        }
      >
        <Scrub m={m} />
      </Figure>

      <H3>Being able to read it off is not the same as it driving the answer</H3>
      <Figure
        label="Experiment 5 — learned detectors (Llama-3.1-8B)."
        caption={
          <>
            We trained a set of small detectors, each tuned to spot one pattern in the notes
            (these come from a tool called a &ldquo;sparse autoencoder&rdquo;). Some of them can
            tell which conclusion the model reached <em>perfectly</em> — a clean 1.0 on the left.
            Yet when we let only those top detectors steer the model, they bring back only about
            half the decision, and the real signal turns out to be spread over roughly ten to
            thirty of them. It's the same lesson as the controls in the previous section, now down
            to the finest level: being able to <em>read</em> something off the notes doesn't mean
            it <em>drives</em> the answer. Judge a note by the effect it has, not by what a detector
            can pull out of it.
          </>
        }
      >
        <Sae />
      </Figure>

      <Aside>
        <b>Why this matters for what comes next.</b> Because the writing is spread all over the
        model, there's no clean knob in the model's wiring to turn. But the thing it writes — the
        note in the notebook — is small, sits in one place, and we can change it directly. Editing
        the note means working at the one natural choke point in this machinery. That's exactly
        what the next two sections do.
      </Aside>
    </Section>
  )
}
