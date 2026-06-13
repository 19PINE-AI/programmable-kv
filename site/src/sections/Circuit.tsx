import { useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH } from '../components/charts/BarCI'
import { ChartSvg, COLORS, Legend } from '../components/charts/core'
import { fmt } from '../lib/format'
import circuit from '../data/circuit.json'

const META = { id: 'circuit', num: '5', title: 'The circuit: distributed write, concentrated read' }

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
        { label: 'read heads (decision → aggregator)', color: COLORS.blue },
        { label: 'write heads (aggregator ← field/rule, at prefill)', color: COLORS.orange },
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
        xLabel="top-k heads patched together"
        yLabel="decision recovery"
        yDomain={[-0.05, 1.0]}
        xTicks={[1, 2, 3, 5, 8, 12]}
        height={280}
        refLinesY={[
          ...(m.read_ctrl ? [{ y: m.read_ctrl.mean, label: `random heads ${fmt(m.read_ctrl.mean, 3)}`, color: COLORS.gray }] : []),
        ]}
      />
      <Legend items={[
        { label: 'cumulative read heads — concentrates', color: COLORS.blue },
        { label: 'cumulative write heads — saturates low (write is distributed)', color: COLORS.orange },
        { label: 'random-head control ≈ 0', color: COLORS.gray, dash: true },
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
          layer (up to the readout layer L{c.readout_layer})
        </text>
        <text className="axis-label" transform={`translate(13,${(H - pad.b + pad.t) / 2}) rotate(-90)`} textAnchor="middle">
          contribution to the note
        </text>
      </ChartSvg>
      <Legend items={[
        { label: `attention writes the note — share ${fmt(c.attn_share?.mean, 2)}`, color: COLORS.orange },
        { label: `MLP — share ${fmt(c.mlp_share?.mean, 2)}`, color: COLORS.purple },
      ]} />
    </div>
  )
}

function Direction({ m }: { m: any }) {
  if (!m.direction) return null
  const layers: number[] = m.direction.layers
  const pl = m.direction.per_layer
  const series = [
    { id: 'full', label: 'full residual patch', color: COLORS.gray, key: 'full', dash: true },
    { id: 'along', label: 'along the 1-D conclusion direction', color: COLORS.blue, key: 'along' },
    { id: 'random', label: 'random 1-D direction', color: COLORS.red, key: 'random', dash: true },
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
        xLabel="layer"
        yLabel="recovery transferred"
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
        { label: 'resample everything ELSE, same conclusion (drift)', value: s.drift?.drift_rest_same?.mean, lo: s.drift?.drift_rest_same?.ci?.[0], hi: s.drift?.drift_rest_same?.ci?.[1], color: COLORS.gray },
        { label: 'resample the NOTE, same conclusion (drift)', value: s.drift?.drift_note_same?.mean, lo: s.drift?.drift_note_same?.ci?.[0], hi: s.drift?.drift_note_same?.ci?.[1], color: COLORS.gray },
        { label: `swap the NOTE to the opposite conclusion (k=${s.k_note})`, value: s.interchange?.rec_note_opp?.mean, lo: s.interchange?.rec_note_opp?.ci?.[0], hi: s.interchange?.rec_note_opp?.ci?.[1], color: COLORS.orange },
        { label: 'swap everything ELSE to the opposite conclusion', value: s.interchange?.rec_rest_opp?.mean, lo: s.interchange?.rec_rest_opp?.ci?.[0], hi: s.interchange?.rec_rest_opp?.ci?.[1], color: COLORS.blue },
      ]}
      domain={[0, 1.0]}
      xLabel="decision movement toward the resampled conclusion"
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
        <div className="fig-sub" style={{ marginBottom: 6 }}>decoding — feature AUC (top features, layer 14 SAE)</div>
        <BarsH
          items={L.top_features_auc.slice(0, 6).map(([feat, auc]: [number, number]) => ({
            label: `feature #${feat}`, value: auc, color: auc >= 0.99 ? COLORS.green : COLORS.blueSoft as any,
          }))}
          domain={[0, 1.05]}
          xLabel="conclusion-decoding AUC"
          labelWidth={110}
          width={420}
        />
      </div>
      <div>
        <div className="fig-sub" style={{ marginBottom: 6 }}>causation — recovery from top-K features alone</div>
        <BarsH
          items={[
            ...L.sufficiency_byK.map((x: any) => ({ label: `top-${x.K} features`, value: x.mean, color: COLORS.orange })),
            { label: 'random features (control)', value: L.control_byK?.[L.control_byK.length - 1]?.mean ?? 0, color: COLORS.gray },
          ]}
          domain={[-0.05, 1.05]}
          xLabel="decision recovery"
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
        Five further interventions resolve the mechanism to <em>components</em>: named attention
        heads, a causal direction, sparse features, and a causal-scrubbing validation — replicated
        across the four families. The shape that emerges everywhere:{' '}
        <strong>attention writes the note redundantly across many mid-layer heads; a small,
        nameable set of late heads reads it into the decision.</strong>
      </P>

      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={models.map((x: any) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
      </Controls>

      <Figure
        label="Exp 1 — the heads."
        title="Read heads concentrate; write heads don't"
        caption={
          <>
            Each dot is a named head (hover for its id and score). Read heads — those attending
            decision→aggregator at decode — are few and individually strong (top head{' '}
            {m.read_heads[0]?.head} alone recovers {fmt(m.read_heads[0]?.rec, 2)} on {m.label});
            write heads at prefill are many and individually weak. Patching random heads recovers{' '}
            {fmt(m.read_ctrl?.mean, 3)}.
          </>
        }
      >
        <HeadMap m={m} />
      </Figure>

      <Figure
        label="Cumulative top-k."
        caption={
          <>
            Stacking the top read heads recovers the decision rapidly (top-12:{' '}
            {fmt(m.read_cumk?.[m.read_cumk.length - 1]?.mean, 2)} on {m.label}); stacking write
            heads saturates low — the write is genuinely distributed, so there is no single
            &ldquo;note-writing head&rdquo; to ablate. This asymmetry is why editing the cache
            beats editing the computation.
          </>
        }
      >
        <CumK m={m} />
      </Figure>

      <Figure
        label="Exp 2 — attention vs. MLP."
        caption={
          <>
            Decomposing what builds the note at the aggregator position, layer by layer up to the
            readout: attention dominates the write ({fmt(m.components?.attn_share?.mean, 2)} share
            on {m.label}) — consistent with a cross-token copy of a computed conclusion rather
            than per-token feature synthesis.
          </>
        }
      >
        <AttnMlp m={m} />
      </Figure>

      <Figure
        label="Exp 3 — a causal conclusion direction."
        caption={
          <>
            A difference-of-means direction in the aggregator&rsquo;s residual stream transfers
            the decision when patched <em>along that single dimension</em> — far above a random
            direction at the same layer — peaking mid-stack exactly where the timing analysis put
            the write. The note has low-rank structure, but (next panels) no single feature is the
            whole story.
          </>
        }
      >
        <Direction m={m} />
      </Figure>

      <Figure
        narrow
        label="Exp 4 — causal scrubbing."
        caption={
          <>
            The clinching test: resample the note from runs with the <em>same</em> conclusion and
            nothing moves (drift ≈ {fmt(m.scrub?.drift?.drift_note_same?.mean, 2)}); swap the note
            for the <em>opposite</em> conclusion and the decision follows it (
            {fmt(m.scrub?.interchange?.rec_note_opp?.mean, 2)}); swap everything <em>except</em>{' '}
            the note and it mostly doesn&rsquo;t ({fmt(m.scrub?.interchange?.rec_rest_opp?.mean, 2)}).
            The note alone governs the decision.
          </>
        }
      >
        <Scrub m={m} />
      </Figure>

      <H3>Decodable ≠ causal, down to single features</H3>
      <Figure
        label="Exp 5 — SAE features (Llama-3.1-8B, layer-14 SAE)."
        caption={
          <>
            A trained sparse autoencoder finds features that decode the conclusion{' '}
            <em>perfectly</em> (AUC 1.0) — yet patching the top features alone recovers only about
            half the decision, spreading over ~10–30 features. The same
            decodability-vs-causation split the controls established in §4, now at the level of
            single features: read the cache by its causal effect, not by what a probe can extract.
          </>
        }
      >
        <Sae />
      </Figure>

      <Aside>
        <b>Why this matters for the capabilities.</b> The write is distributed (no clean
        weight-space handle), but the <em>written artifact</em> — the note in the KV cache — is
        compact, localized, and writable. Intervening on the cache is intervening at the
        circuit&rsquo;s natural bottleneck. That is exactly what the next two sections do.
      </Aside>
    </Section>
  )
}
