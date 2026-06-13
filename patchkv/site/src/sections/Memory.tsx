import { useState } from 'react'
import { Section, P, H3, Aside, PaperConst } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg, ModelPicker } from '../components/ui/Controls'
import { LineChart } from '../components/charts/LineChart'
import { BarsH } from '../components/charts/BarCI'
import { Heatmap, ramp } from '../components/charts/Heatmap'
import { ChartSvg, COLORS, Legend } from '../components/charts/core'
import { fmt, fmtX } from '../lib/format'
import memory from '../data/memory.json'
import constants from '../data/constants.json'

const META = { id: 'memory', num: '11', title: 'Application: editable and composable user memory' }

/* ---------------- placement dilemma schematic ---------------- */

function PlacementDiagram() {
  const W = 720
  const layouts = [
    { name: 'memory at the FRONT', cells: ['sys', 'MEMORY', 'trajectory…', 'query'], memIdx: 1, note: 'trajectory memoizes memory-conditioned conclusions → a change reprefills everything after' },
    { name: 'memory at the END', cells: ['sys', 'trajectory…', 'MEMORY', 'query'], memIdx: 2, note: 'memory’s KV depends on the trajectory → re-attended every turn' },
    { name: 'this paper: compose + edit', cells: ['sys', 'trajectory…', 'MEMORY ⟲', 'query'], memIdx: 2, note: 'precompiled once, RoPE-repositioned each turn, 1 seam token repaired, edited in place', win: true },
  ]
  return (
    <ChartSvg width={W} height={layouts.length * 64 + 8}>
      {layouts.map((l, i) => {
        const y = 6 + i * 64
        let x = 200
        return (
          <g key={l.name}>
            <text x={192} y={y + 20} textAnchor="end" style={{ fontFamily: 'var(--sans)', fontSize: 11.5, fontWeight: 600 }}
              fill={l.win ? COLORS.green : 'var(--ink-soft)'}>
              {l.name}
            </text>
            {l.cells.map((c, j) => {
              const w = c === 'trajectory…' ? 130 : c.startsWith('MEMORY') ? 110 : 52
              const el = (
                <g key={j}>
                  <rect x={x} y={y} width={w} height={30} rx={5}
                    fill={j === l.memIdx ? 'var(--orange-faint)' : '#fff'}
                    stroke={j === l.memIdx ? COLORS.orange : 'var(--rule-strong)'} strokeWidth={j === l.memIdx ? 1.6 : 1} />
                  <text x={x + w / 2} y={y + 19} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontSize: 10 }} fill="var(--ink-soft)">{c}</text>
                </g>
              )
              x += w + 8
              return el
            })}
            <text x={200} y={y + 45} style={{ fontFamily: 'var(--sans)', fontSize: 10.5 }} fill="var(--ink-faint)">{l.note}</text>
          </g>
        )
      })}
    </ChartSvg>
  )
}

/* ---------------- E1: placement cost grows with length ---------------- */

function E1Length() {
  const bl = (memory.e1.accuracy_by_len as any)['Qwen/Qwen3-4B|cot']
  const MT = [24, 120, 880, 1700]
  const MT_LABEL: Record<number, string> = { 24: '24 facts (~0.5k tok)', 120: '120 (~2k)', 880: '880 (~16k)', 1700: '1700 (~32k)' }
  const pts = MT.map((mt) => {
    const nf = mt >= 880 ? 'nf8' : 'nf1'
    const e = bl[`early_${nf}_mt${mt}`]
    const l = bl[`late_${nf}_mt${mt}`]
    return e && l ? { mt, gap: e.acc - l.acc, early: e.acc, late: l.acc } : null
  }).filter(Boolean) as any[]

  return (
    <div>
      <LineChart
        series={[
          { id: 'early', color: COLORS.blue, points: pts.map((p, i) => ({ x: i, y: p.early })) },
          { id: 'late', color: COLORS.orange, points: pts.map((p, i) => ({ x: i, y: p.late })) },
        ]}
        xTicks={pts.map((_, i) => i)}
        xFmt={(i) => MT_LABEL[pts[Math.round(i)]?.mt] ?? ''}
        xLabel="memory size"
        yLabel="decision accuracy (full recompute, CoT)"
        yDomain={[0.5, 1.05]}
        height={270}
      />
      <Legend items={[
        { label: 'memory read EARLY (pre-digested at prefill)', color: COLORS.blue },
        { label: 'memory read LATE (raw at decode)', color: COLORS.orange },
      ]} />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        early−late gap: {pts.map((p) => `${p.gap >= 0 ? '+' : ''}${fmt(p.gap, 2)}`).join(' → ')} as memory grows
        (Qwen3-4B, GEE-logistic β<sub>late</sub> = {fmt((memory.e1.placement_gee as any)['Qwen3-4B'].terms['C(placement)[T.late]'].coef, 2)},
        p = {(memory.e1.placement_gee as any)['Qwen3-4B'].terms['C(placement)[T.late]'].p})
      </div>
    </div>
  )
}

/* ---------------- E2: seam dose-response ---------------- */

function E2Seam() {
  const rows = memory.e2.seam as any[]
  const labels = [...new Set(rows.map((r) => r.label))]
  const [label, setLabel] = useState('Llama-3.1-70B (4-bit)')
  const sel = labels.includes(label) ? label : labels[0]
  const early = rows.find((r) => r.label === sel && r.placement === 'early')
  const late = rows.find((r) => r.label === sel && r.placement === 'late')
  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={labels.map((l) => ({ id: l, label: l }))} value={sel} onChange={setLabel} />
        </ControlGroup>
      </Controls>
      <LineChart
        series={[
          ...(late ? [{ id: 'late', color: COLORS.orange, points: late.doses.map((d: any) => ({ x: d.seam, y: d.dec_agree, lo: d.dec_agree_lo })), band: false }] : []),
          ...(early ? [{ id: 'early', color: COLORS.blue, dash: true, points: early.doses.map((d: any) => ({ x: d.seam, y: d.dec_agree, lo: d.dec_agree_lo })) }] : []),
        ]}
        xTicks={[0, 1, 2, 4, 8]}
        xLabel="seam-repair tokens recomputed at the chunk boundary"
        yLabel="decision agreement vs. full recompute"
        yDomain={[0.4, 1.02]}
        height={270}
      />
      <Legend items={[
        { label: 'late placement (decode reads memory directly)', color: COLORS.orange },
        { label: 'early placement (reads through pre-digested notes)', color: COLORS.blue, dash: true },
      ]} />
      {sel.includes('70B') && late && early && (
        <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
          The cleanest decision-governance test: 70B&rsquo;s decisions genuinely vary, and{' '}
          <b>late beats early ({fmt(late.doses[2].dec_agree, 2)} vs {fmt(early.doses[2].dec_agree, 2)} at seam 2)</b>{' '}
          exactly as the mechanism predicts. A single seam token closes most of the boundary gap.
        </div>
      )}
    </div>
  )
}

/* ---------------- E3: edit modes ---------------- */

function E3Heat() {
  const by = memory.e3.by_model as any
  const models = Object.keys(by)
  const methods = ['stale', 'in_place', 'erratum', 'recompile_chunk', 'selective@16', 'full_recompute']
  const LBL: Record<string, string> = {
    stale: 'stale', in_place: 'in-place (1 tok)', erratum: 'erratum',
    recompile_chunk: 'recompile chunk', 'selective@16': 'selective@16', full_recompute: 'full',
  }
  return (
    <div>
      <Heatmap
        rows={models}
        cols={methods.map((m) => LBL[m])}
        value={(r, c) => by[models[r]][methods[c]]?.correct ?? null}
        colorOf={(v) => (v === null ? '#f0eee6' : ramp(v, [74, 138, 92]))}
        colLabel="P(correct flipped decision) after a mid-session memory edit (CoT)"
        rowLabelWidth={140}
        tooltip={(r, c) => {
          const cell = by[models[r]][methods[c]]
          return cell ? `${models[r]} · ${methods[c]}: correct ${fmt(cell.correct, 2)}, ~${cell.recompute_tok} tokens recomputed` : null
        }}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        The near-free in-place edit (one token recomputed) strengthens with scale:{' '}
        {Object.entries(memory.e3.scale_inplace as any)
          .filter(([k]) => k.includes('Qwen'))
          .map(([k, v]) => `${k.split('/').pop()} ${fmt(v as number, 2)}`)
          .join(' · ')}
        . Where the chain does not re-read the field (Llama-3.1-8B,{' '}
        {fmt((memory.e3.scale_inplace as any)['unsloth/Meta-Llama-3.1-8B-Instruct'], 2)}), the
        append-only erratum is the robust fallback (McNemar p ={' '}
        {(by['Llama-3.1-8B']?.['_mcnemar_inplace_vs_erratum']?.p ?? 0.031).toFixed(3)}).
      </div>
    </div>
  )
}

/* ---------------- keystone inside memory (70B) ---------------- */

function Keystone70() {
  const k70 = Object.values(memory.keystone70 as any)[0] as any
  const order = ['stale', 'in_place', 'selective@4', 'selective@16', 'recompile_chunk', 'full_recompute']
  const items = order.filter((m) => k70[m]).map((m) => ({
    label: m, value: k70[m].correct, lo: k70[m].correct_lo, color: m === 'in_place' ? COLORS.orange : m.startsWith('select') ? COLORS.blue : m === 'full_recompute' ? COLORS.gray : COLORS.green,
  }))
  return (
    <BarsH
      items={items}
      domain={[0, 1.05]}
      xLabel="P(correct) editing a field INSIDE a transplanted memory chunk — Llama-3.1-70B, direct mode"
      labelWidth={150}
    />
  )
}

/* ---------------- E4 + xref: granularity and the block-split test ---------------- */

function BlockSplit() {
  const xref = memory.xref as any[]
  const [tag, setTag] = useState('llama31_8b')
  const [split, setSplit] = useState(true)
  const m = xref.find((x) => x.tag === tag)!
  const s = m.summary
  const agree = split ? s.xref.split_agree : s.xref.colo_agree
  const indep = split ? s.indep.split_agree : s.indep.colo_agree

  const W = 660
  return (
    <div>
      <Controls>
        <ControlGroup label="model">
          <ModelPicker models={xref.map((x) => ({ id: x.tag, label: x.label }))} value={tag} onChange={setTag} />
        </ControlGroup>
        <ControlGroup label="block boundary">
          <Seg options={['split', 'colocated'] as const} value={split ? 'split' : 'colocated'}
            onChange={(v) => setSplit(v === 'split')} accent="orange"
            labels={{ split: 'between A and B', colocated: 'past both' }} />
        </ControlGroup>
      </Controls>

      <ChartSvg width={W} height={150}>
        {/* memory as two precompiled blocks */}
        {(() => {
          const bx = split ? 312 : 432
          return (
            <g>
              <text x={20} y={22} style={{ fontFamily: 'var(--sans)', fontSize: 11.5, fontWeight: 600 }} fill="var(--ink-soft)">
                memory, precompiled as independent blocks:
              </text>
              <rect x={20} y={36} width={bx - 24} height={44} rx={6} fill="var(--blue-faint)" stroke={COLORS.blue} />
              <rect x={bx + 4} y={36} width={620 - bx} height={44} rx={6} fill="var(--blue-faint)" stroke={COLORS.blue} />
              {/* facts */}
              <text x={120} y={62} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700 }} fill={COLORS.orange}>
                A: DEFINITION → “gate on setting_X”
              </text>
              <text x={500} y={62} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700 }} fill={COLORS.orange}>
                B: setting_X = false
              </text>
              <line x1={bx} x2={bx} y1={28} y2={92} stroke={COLORS.red} strokeWidth={2.5} strokeDasharray="5 3" />
              <text x={bx} y={108} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 10.5, fontWeight: 700 }} fill={COLORS.red}>
                block boundary {split ? '— splits the A→B chain' : '— A and B stay together'}
              </text>
              <text x={20} y={140} style={{ fontFamily: 'var(--sans)', fontSize: 11 }} fill="var(--ink-faint)">
                block B precompiled in isolation never attended to its referent A — split the pair and the two-hop chain is never memoized
              </text>
            </g>
          )
        })()}
      </ChartSvg>

      <div style={{ display: 'flex', gap: 28, fontFamily: 'var(--sans)', fontSize: 13, marginTop: 8 }}>
        <span>
          cross-referential pair A→B: agreement{' '}
          <b style={{ color: agree < 0.6 ? 'var(--red)' : 'var(--green)', fontSize: 16 }}>{fmt(agree, 2)}</b>
        </span>
        <span>
          independent pair (control): <b style={{ fontSize: 16 }}>{fmt(indep, 2)}</b>
        </span>
        <span style={{ color: 'var(--ink-faint)' }}>n={s.xref.n}, full-recompute accuracy {fmt(s.xref.full_acc, 2)}</span>
      </div>
    </div>
  )
}

function E4Sweep() {
  const by = memory.e4.by_model as any
  const models = Object.keys(by)
  return (
    <div>
      <LineChart
        series={models.map((m, i) => ({
          id: m, color: [COLORS.blue, COLORS.purple][i % 2],
          points: by[m].map((row: any) => ({ x: row.S, y: row.dec_agree ?? row.top1_agree })),
        }))}
        xTicks={[1, 2, 4, 8, 16]}
        xLog
        xLabel="memory split into S independently-precompiled blocks"
        yLabel="decision agreement vs. S=1"
        yDomain={[0.5, 1.05]}
        height={250}
      />
      <Legend items={models.map((m, i) => ({ label: m, color: [COLORS.blue, COLORS.purple][i % 2] }))} />
    </div>
  )
}

/* ---------------- E5 agent + LoCoMo ---------------- */

function E5Bars() {
  const by = memory.e5.by_model as any
  const models = Object.keys(by)
  return (
    <BarsH
      items={models.map((m) => ({
        label: `${m} (${by[m].n_sessions} sessions)`,
        value: by[m].cum_speedup_vs_end,
        color: COLORS.orange,
        note: `· cos ${fmt(by[m].proposed_cos, 2)}`,
      }))}
      domain={[0, Math.max(...models.map((m) => by[m].cum_speedup_vs_end)) * 1.2]}
      xLabel="cumulative TTFT speedup vs. reprefill-every-turn-at-the-end (note: logit cosine vs. token-matched oracle)"
      valueFmt={(v) => fmtX(v, 1)}
      labelWidth={200}
    />
  )
}

function LocomoTable() {
  const lc = memory.locomo as any
  const models = Object.keys(lc)
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>model</th><th>full recompute</th><th>transplanted memory</th><th>Δ accuracy</th>
          <th>TOST equivalent (±3pt)</th><th>answer-logit cos</th>
        </tr>
      </thead>
      <tbody>
        {models.map((m) => {
          const r = lc[m]
          return (
            <tr key={m}>
              <td style={{ fontWeight: 600 }}>{m}</td>
              <td>{fmt(r.acc_full, 3)}</td>
              <td>{fmt(r.acc_transplant, 3)}</td>
              <td>{r.acc_parity_diff >= 0 ? '+' : ''}{fmt(r.acc_parity_diff, 3)}</td>
              <td style={{ color: r.acc_parity_equivalent ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>
                {r.acc_parity_equivalent ? '✓ equivalent' : '✗ small deficit'}
              </td>
              <td>{fmt(r.ans_cos, 3)}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export function Memory() {
  return (
    <Section meta={META}>
      <P>
        The highest-value instance of the edit+compose substrate is <strong>user memory</strong>:
        the large, dynamically summarized profile an assistant re-reads every turn. Memory is big
        (10³–10⁴ tokens), reused across turns, and mutated mid-session by tool calls — so where it
        lives in the prompt is a dilemma the mechanism explains precisely:
      </P>

      <Figure narrow label="The placement dilemma." caption={<>Front placement pre-digests but makes every memory change expensive; end placement is cheap to change but re-attends memory every turn. The paper&rsquo;s resolution treats memory as a <em>skill that is also edited</em>.</>}>
        <PlacementDiagram />
      </Figure>

      <H3>E1 — pre-digestion is real, and it is the price of late placement</H3>
      <Figure
        label="Placement × memory length."
        caption={
          <>
            Under full recompute and CoT, reading memory late costs a small but statistically
            significant amount of accuracy — and the cost grows with memory length. Late placement
            is still preferred: it buys O(L) editing/transplant and the TTFT wins below, a
            tradeoff the paper states rather than hides (early placement is preferable when memory
            is very long and accuracy-critical).
          </>
        }
      >
        <E1Length />
      </Figure>

      <H3>E2 — memory transplant is faithful, to 70B</H3>
      <Figure
        label="Seam dose-response."
        caption={
          <>
            A precompiled, RoPE-repositioned memory chunk reproduces full-recompute decisions
            across ten models (cosine 0.94–0.9996); one seam-repair token closes the
            start-of-chunk boundary. A no-rotation control collapses (agreement 0.18 vs 0.78 on
            70B) — the re-rotation is what carries it.
          </>
        }
      >
        <E2Seam />
      </Figure>

      <H3>E3 — memory is editable mid-session</H3>
      <Figure
        label="Edit modes."
        caption={
          <>
            When a stored fact toggles, reusing stale memory recovers the flipped decision
            essentially never (≤0.03); every real edit recovers it. Consistent with §6, the
            near-free in-place edit suffices under CoT and strengthens with scale.
          </>
        }
      >
        <E3Heat />
      </Figure>

      <Figure
        narrow
        label="Keystone, again — inside transplanted memory."
        caption={
          <>
            Editing a field <em>inside a transplanted memory chunk</em> (70B, direct mode, where
            decisions genuinely vary) reproduces §2&rsquo;s memoization verbatim: the
            field&rsquo;s own KV recovers little, recovery climbs with selective recompute, and
            under CoT the stickiness dissolves (a flat selective@K sweep at ≈0.98). One substrate,
            twice over.
          </>
        }
      >
        <Keystone70 />
      </Figure>

      <H3>E4 — granularity is a free knob, with one sharp edge</H3>
      <Figure
        label="Block granularity."
        caption={
          <>
            Splitting memory into S independently-precompiled blocks makes a localized edit
            S× cheaper and stays decision-lossless to S=16 — independent facts are integrated at
            read time. The sharp edge is <em>cross-referential</em> facts:
          </>
        }
      >
        <E4Sweep />
      </Figure>

      <Figure
        label="The block-split test."
        caption={
          <>
            A decision needing a two-hop chain (a DEFINITION names which setting gates; the value
            lives elsewhere): splitting the linked pair across a block boundary drops agreement to
            0.46 vs 0.76 colocated (Llama-3.1-8B, McNemar p&lt;10⁻⁶) — while the same split costs
            an independent pair almost nothing. Practical guidance: keep cross-referential facts
            in one block.
          </>
        }
      >
        <BlockSplit />
      </Figure>

      <H3>E5 — the live memory agent, and real conversations</H3>
      <Figure
        narrow
        label="End-to-end agent."
        caption={
          <>
            A live agent that composes memory once, re-rotates it each turn, and edits it on
            tool-driven changes: 2.3–4.3× lower cumulative TTFT vs. reprefill-every-turn, while
            reproducing full-reprefill next-token logits at a token-matched oracle. One honest
            caveat: greedy CoT chains are boundary-sensitive, so exact chains reproduce only{' '}
            {constants.e5_chain_agreement.range[0]}–{constants.e5_chain_agreement.range[1]} of the
            time even though decisions are faithful.{' '}
            <PaperConst src={constants.e5_chain_agreement.source} />
          </>
        }
      >
        <E5Bars />
      </Figure>

      <Figure
        narrow
        label="LoCoMo — external validity on real long conversations."
        caption={
          <>
            The multi-session LoCoMo dialogues (median ~19.7k tokens) as the memory, precompiled
            and spliced before each of all 1,540 answerable questions per model: transplant is
            statistically equivalent to full recompute (TOST, ±3pt margin) on the three Qwen3
            models, and within −2.7 points on Llama-3.1-8B — shown, not smoothed over.
          </>
        }
      >
        <LocomoTable />
      </Figure>
    </Section>
  )
}
