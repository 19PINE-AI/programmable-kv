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

const META = { id: 'memory', num: '4', title: 'A living memory' }

/* ---------------- placement dilemma schematic ---------------- */

function PlacementDiagram() {
  const W = 720
  const layouts = [
    { name: 'memory at the START', cells: ['sys', 'MEMORY', 'conversation…', 'query'], memIdx: 1, note: 'the rest of the chat bakes in what it read from memory → change one fact and everything after must be reread' },
    { name: 'memory at the END', cells: ['sys', 'conversation…', 'MEMORY', 'query'], memIdx: 2, note: 'memory now depends on the chat before it → it gets reread from scratch every turn' },
    { name: 'this paper: reuse + edit', cells: ['sys', 'conversation…', 'MEMORY ⟲', 'query'], memIdx: 2, note: 'read once and saved, dropped back in each turn, one boundary note fixed up, edited in place when a fact changes', win: true },
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
        xLabel="size of memory"
        yLabel="answer accuracy (rereading everything)"
        yDomain={[0.5, 1.05]}
        height={270}
      />
      <Legend items={[
        { label: 'memory placed EARLY (model digests it before the chat)', color: COLORS.blue },
        { label: 'memory placed LATE (model reads it right before answering)', color: COLORS.orange },
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
        xLabel="number of words recomputed at the start of the pasted-in memory"
        yLabel="how often the answer matches rereading everything"
        yDomain={[0.4, 1.02]}
        height={270}
      />
      <Legend items={[
        { label: 'memory placed late (read right before answering)', color: COLORS.orange },
        { label: 'memory placed early (digested before the chat)', color: COLORS.blue, dash: true },
      ]} />
      {sel.includes('70B') && late && early && (
        <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
          This is the clearest test, because the 70B model&rsquo;s answers really do change. And{' '}
          <b>late placement beats early ({fmt(late.doses[2].dec_agree, 2)} vs {fmt(early.doses[2].dec_agree, 2)} with 2 words fixed up)</b>{' '}
          exactly as expected. Recomputing a single word at the boundary closes most of the gap.
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
    stale: 'no edit', in_place: 'edit in place (1 word)', erratum: 'append a correction',
    recompile_chunk: 'reread memory', 'selective@16': 'reread 16 words', full_recompute: 'reread all',
  }
  return (
    <div>
      <Heatmap
        rows={models}
        cols={methods.map((m) => LBL[m])}
        value={(r, c) => by[models[r]][methods[c]]?.correct ?? null}
        colorOf={(v) => (v === null ? '#f0eee6' : ramp(v, [74, 138, 92]))}
        colLabel="how often the model gives the right answer after a fact changes mid-conversation"
        rowLabelWidth={140}
        tooltip={(r, c) => {
          const cell = by[models[r]][methods[c]]
          return cell ? `${models[r]} · ${methods[c]}: correct ${fmt(cell.correct, 2)}, ~${cell.recompute_tok} words recomputed` : null
        }}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        The nearly-free in-place edit (just one word recomputed) works even better in bigger models:{' '}
        {Object.entries(memory.e3.scale_inplace as any)
          .filter(([k]) => k.includes('Qwen'))
          .map(([k, v]) => `${k.split('/').pop()} ${fmt(v as number, 2)}`)
          .join(' · ')}
        . When the model doesn&rsquo;t re-read the changed fact (Llama-3.1-8B,{' '}
        {fmt((memory.e3.scale_inplace as any)['unsloth/Meta-Llama-3.1-8B-Instruct'], 2)}), simply
        appending a correction is a reliable fallback (p ={' '}
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
      xLabel="how often the answer is right after editing a fact INSIDE the pasted-in memory — Llama-3.1-70B"
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
            labels={{ split: 'between A and B', colocated: 'keep A and B together' }} />
        </ControlGroup>
      </Controls>

      <ChartSvg width={W} height={150}>
        {/* memory as two precompiled blocks */}
        {(() => {
          const bx = split ? 312 : 432
          return (
            <g>
              <text x={20} y={22} style={{ fontFamily: 'var(--sans)', fontSize: 11.5, fontWeight: 600 }} fill="var(--ink-soft)">
                memory, saved as separate blocks:
              </text>
              <rect x={20} y={36} width={bx - 24} height={44} rx={6} fill="var(--blue-faint)" stroke={COLORS.blue} />
              <rect x={bx + 4} y={36} width={620 - bx} height={44} rx={6} fill="var(--blue-faint)" stroke={COLORS.blue} />
              {/* facts */}
              <text x={120} y={62} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700 }} fill={COLORS.orange}>
                A: rule → “depends on setting_X”
              </text>
              <text x={500} y={62} textAnchor="middle" style={{ fontFamily: 'var(--mono)', fontSize: 10.5, fontWeight: 700 }} fill={COLORS.orange}>
                B: setting_X = false
              </text>
              <line x1={bx} x2={bx} y1={28} y2={92} stroke={COLORS.red} strokeWidth={2.5} strokeDasharray="5 3" />
              <text x={bx} y={108} textAnchor="middle" style={{ fontFamily: 'var(--sans)', fontSize: 10.5, fontWeight: 700 }} fill={COLORS.red}>
                block boundary {split ? '— separates the linked A and B' : '— A and B stay together'}
              </text>
              <text x={20} y={140} style={{ fontFamily: 'var(--sans)', fontSize: 11 }} fill="var(--ink-faint)">
                a block saved on its own never saw the fact it depends on — split a linked pair and the model can&rsquo;t connect them
              </text>
            </g>
          )
        })()}
      </ChartSvg>

      <div style={{ display: 'flex', gap: 28, fontFamily: 'var(--sans)', fontSize: 13, marginTop: 8 }}>
        <span>
          two linked facts (A needs B): match{' '}
          <b style={{ color: agree < 0.6 ? 'var(--red)' : 'var(--green)', fontSize: 16 }}>{fmt(agree, 2)}</b>
        </span>
        <span>
          two unrelated facts (control): <b style={{ fontSize: 16 }}>{fmt(indep, 2)}</b>
        </span>
        <span style={{ color: 'var(--ink-faint)' }}>n={s.xref.n}, accuracy when rereading all {fmt(s.xref.full_acc, 2)}</span>
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
        xLabel="memory split into S separately-saved blocks"
        yLabel="how often the answer matches a single block"
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
      xLabel="how much faster the assistant starts replying vs. rereading memory every turn (response-latency speedup)"
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
          <th>model</th><th>reread everything</th><th>reused memory</th><th>Δ accuracy</th>
          <th>statistically equivalent (±3 pts)</th><th>answer similarity</th>
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
        Here is where the idea really pays off: an assistant&rsquo;s <strong>memory</strong>. This
        is a long, growing profile of facts about you that the assistant re-reads at the start of
        every turn. Re-reading it each time is slow, because the model first turns all that text
        into internal notes (think of it as the model&rsquo;s private notebook of what it just
        read). Our idea: compute those notes once, drop them back in each turn instead of
        re-reading, and edit them in place when a single fact changes. The catch is where memory
        sits in the conversation — and that choice forces a trade-off:
      </P>

      <Figure narrow label="Where should memory go?" caption={<>Put memory at the start and the model digests it up front, but then any change to memory is expensive. Put it at the end and changes are cheap, but the model re-reads memory every turn. Our fix: treat memory as a <em>reusable note that can also be edited</em>.</>}>
        <PlacementDiagram />
      </Figure>

      <H3>E1 — reading memory early helps a little, and the gap grows with length</H3>
      <Figure
        label="Where memory sits vs. how long it is."
        caption={
          <>
            Even when the model rereads everything, placing memory late costs a small but real bit
            of accuracy — and that cost grows as memory gets longer. We still prefer late
            placement: it makes editing and reuse cheap, and it gives the big speed wins shown
            below. We state this trade-off plainly rather than hide it (early placement is the
            better choice when memory is very long and accuracy is critical).
          </>
        }
      >
        <E1Length />
      </Figure>

      <H3>E2 — pasting in saved memory gives the same answers, even at 70B</H3>
      <Figure
        label="Fixing up the boundary."
        caption={
          <>
            Saving memory once and pasting it back in (after correcting for its new position in
            the prompt) reproduces the same decisions as rereading everything, across ten models
            (answers 94–99.96% similar). Recomputing just one word at the start of the pasted-in
            block cleans up the boundary. Skip the position correction and it falls apart (matches
            18% of the time instead of 78% on the 70B model) — that correction is what makes it
            work.
          </>
        }
      >
        <E2Seam />
      </Figure>

      <H3>E3 — you can edit memory mid-conversation</H3>
      <Figure
        label="Ways to update a changed fact."
        caption={
          <>
            When a stored fact flips, reusing the old memory almost never gives the updated answer
            (3% or less); every real edit does. The nearly-free in-place edit — recomputing just
            one word — is enough, and it works even better in bigger models.
          </>
        }
      >
        <E3Heat />
      </Figure>

      <Figure
        narrow
        label="The same pattern, inside memory."
        caption={
          <>
            Editing a fact <em>inside the pasted-in memory</em> (70B, where the answers genuinely
            change) behaves exactly like elsewhere in the paper: editing just that fact recovers
            little on its own, recovery climbs as you recompute a bit more around it, and when the
            model reasons step by step the stickiness disappears (a flat line at about 0.98). The
            same mechanism, showing up twice.
          </>
        }
      >
        <Keystone70 />
      </Figure>

      <H3>E4 — splitting memory into blocks is mostly free, with one catch</H3>
      <Figure
        label="How finely to split memory."
        caption={
          <>
            Splitting memory into S separately-saved blocks makes editing one spot S times cheaper,
            and the answers stay just as good all the way to 16 blocks — the model stitches
            unrelated facts together when it reads. The one catch is <em>linked</em> facts that
            point to each other:
          </>
        }
      >
        <E4Sweep />
      </Figure>

      <Figure
        label="Splitting linked facts apart."
        caption={
          <>
            Some answers need two facts that point to each other (one fact says which setting
            matters; the setting&rsquo;s value lives elsewhere). Put those two in different blocks
            and the answers match only 46% of the time, versus 76% when they share a block
            (Llama-3.1-8B, p&lt;10⁻⁶) — yet the same split barely hurts two unrelated facts. The
            practical rule: keep linked facts together in one block.
          </>
        }
      >
        <BlockSplit />
      </Figure>

      <H3>E5 — a working assistant, and real conversations</H3>
      <Figure
        narrow
        label="The whole thing, end to end."
        caption={
          <>
            A working assistant that computes its memory once, drops it back in each turn, and
            edits it whenever a fact changes: it starts replying 2.3–4.3× faster than rereading
            memory every turn, while producing the same next word. One honest caveat: when the
            model reasons step by step, its exact wording is sensitive to where the boundary
            falls, so the full chains match only{' '}
            {constants.e5_chain_agreement.range[0]}–{constants.e5_chain_agreement.range[1]} of the
            time — even though the final answers stay correct.{' '}
            <PaperConst src={constants.e5_chain_agreement.source} />
          </>
        }
      >
        <E5Bars />
      </Figure>

      <Figure
        narrow
        label="A real long-conversation benchmark."
        caption={
          <>
            We tested this on LoCoMo, a public benchmark of long, multi-session conversations
            (each around 19.7k words). Using those conversations as the memory and pasting it in
            before each of all 1,540 answerable questions per model, reusing memory was
            statistically just as accurate as rereading everything (within a ±3-point margin) on
            the three Qwen3 models, and within 2.7 points on Llama-3.1-8B — shown plainly, not
            glossed over.
          </>
        }
      >
        <LocomoTable />
      </Figure>
    </Section>
  )
}
