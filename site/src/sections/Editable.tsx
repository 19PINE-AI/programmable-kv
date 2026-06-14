import { useState } from 'react'
import { scaleLog, scaleLinear } from 'd3-scale'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg, ModelPicker } from '../components/ui/Controls'
import { Heatmap, ramp } from '../components/charts/Heatmap'
import { BarsH, BarsV } from '../components/charts/BarCI'
import { AxisBottom, AxisLeft, ChartSvg, COLORS } from '../components/charts/core'
import { fmt, fmtPct, fmtMs } from '../lib/format'
import editing from '../data/editing.json'
import prompts from '../data/prompts.json'
import mechanism from '../data/mechanism.json'

const META = { id: 'editable', num: '3', title: 'Change a fact, skip the redo' }

function ReasoningGap() {
  const dv = (mechanism.diverse as any[]).find((x) => x.tag === 'qwen3_8b')!
  const nonr = dv.modes.nonreasoning
  const reas = dv.modes.reasoning
  return (
    <BarsV
      groups={[
        { label: 'just refresh the fact', values: [{ v: nonr.field_only.P_correct }, { v: reas.field_only.P_correct }] },
        { label: 'redo affected notes', values: [{ v: 1.0 }, { v: 1.0 }] },
        { label: 'correction note', values: [{ v: nonr.erratum.P_correct }, { v: reas.erratum.P_correct }] },
      ]}
      seriesLabels={['answers directly', 'thinks step by step']}
      colors={[COLORS.gray, COLORS.blue]}
      yDomain={[0, 1.12]}
      yLabel="chance of the right decision"
      height={250}
    />
  )
}

const METHOD_INFO: Record<string, { label: string; desc: string; color: string }> = {
  full_reprefill: { label: 'read everything again', color: COLORS.gray, desc: 'have the model re-read the whole prompt — always right, but slow, since the work grows fast as the prompt gets longer' },
  hoist_to_end: { label: 'move the fact to the end', color: COLORS.purple, desc: 'reword the prompt so the changeable fact sits at the very end — cheapest, but you have to rewrite the prompt and know in advance which facts can change' },
  'field+erratum': { label: 'fact + correction note', color: COLORS.green, desc: 'redo the notes for just the changed fact, then add one clear correction note — the paper’s reliable default, and no prompt rewriting needed' },
  erratum: { label: 'correction note only', color: COLORS.green, desc: 'just add the one-line correction note and leave the rest of the notes as they were — you amend the notes instead of redoing them' },
  'cacheblend@15%': { label: 'CacheBlend', color: COLORS.blue, desc: 'redo the notes that changed the most — but it follows the changed wording, not the notes that already wrote down the conclusion, so it fails here' },
  in_place: { label: 'just refresh the fact', color: COLORS.orange, desc: 'redo the notes for only the changed fact (about 1% of the work) — almost free, but recovers nothing unless the model reasons step by step' },
  stale: { label: 'do nothing', color: COLORS.red, desc: 'reuse all the old notes unchanged — the do-nothing starting point' },
}

function Frontier() {
  const methods = (editing.baseline.methods as any[]).filter((m) => METHOD_INFO[m.method])
  const [sel, setSel] = useState('field+erratum')
  const W = 660
  const H = 320
  const pad = { l: 56, r: 20, t: 16, b: 46 }
  const x = scaleLog().domain([0.004, 1.4]).range([pad.l, W - pad.r])
  const y = scaleLinear().domain([-0.04, 1.06]).range([H - pad.b, pad.t])
  const cur = methods.find((m) => m.method === sel)!
  const info = METHOD_INFO[sel]

  return (
    <div>
      <Controls>
        <ControlGroup label="method">
          <Seg
            options={methods.map((m) => m.method) as any}
            value={sel}
            onChange={(v) => setSel(v as string)}
            labels={Object.fromEntries(methods.map((m) => [m.method, METHOD_INFO[m.method].label])) as any}
          />
        </ControlGroup>
      </Controls>

      <ChartSvg width={W} height={H}>
        <AxisBottom scale={x as any} y={H - pad.b} ticks={[0.005, 0.01, 0.05, 0.1, 0.5, 1]} fmt={(v) => fmtPct(v, v < 0.01 ? 1 : 0)} label="share of the prompt redone (log scale)" grid gridY1={pad.t} gridY2={H - pad.b} />
        <AxisLeft scale={y as any} x={pad.l} ticks={[0, 0.25, 0.5, 0.75, 1]} label="chance of the right decision" grid gridX2={W - pad.r} />
        {methods.map((m) => {
          const fr = Math.max(m.recompute_frac, 0.005)
          const isSel = m.method === sel
          return (
            <g key={m.method} onClick={() => setSel(m.method)} style={{ cursor: 'pointer' }}>
              <circle cx={x(fr)} cy={y(m.P_correct)} r={isSel ? 10 : 7}
                fill={METHOD_INFO[m.method].color} opacity={isSel ? 1 : 0.7}
                stroke={isSel ? 'var(--ink)' : '#fff'} strokeWidth={isSel ? 2 : 1.2} />
              <text className="tick-label" x={x(fr)} y={y(m.P_correct) + (m.P_correct > 0.5 ? 24 : -14)}
                textAnchor="middle" style={{ fontWeight: isSel ? 700 : 400, fill: isSel ? 'var(--ink)' : undefined }}>
                {METHOD_INFO[m.method].label}
              </text>
            </g>
          )
        })}
      </ChartSvg>

      <div className="aside" style={{ marginTop: 4 }}>
        <b>{info.label}</b> — right {fmt(cur.P_correct, 2)} of the time, redoing {fmtPct(cur.recompute_frac, 1)} of the prompt.{' '}
        {info.desc}
        {(sel === 'erratum' || sel === 'field+erratum') && (
          <div className="mono" style={{ background: '#fff', borderRadius: 4, padding: '6px 10px', marginTop: 8, fontSize: 11.5 }}>
            {prompts.scenarios[0].erratum_line}
          </div>
        )}
      </div>
    </div>
  )
}

function KsweepHeat() {
  const ks = editing.ksweep as any[]
  const Ks = ks[0].Ks.map((x: any) => x.K)
  return (
    <div>
      <Heatmap
        rows={ks.map((m) => m.label)}
        cols={Ks.map((k: number) => (k === 0 ? 'fact only' : `+top ${k}`))}
        value={(r, c) => ks[r].Ks[c]?.P_correct ?? null}
        colorOf={(v) => (v === null ? '#f0eee6' : ramp(v, [74, 138, 92]))}
        colLabel="redo the fact plus its K most important affected notes (model thinking step by step)"
        rowLabelWidth={160}
        tooltip={(r, c) => `${ks[r].label}, K=${Ks[c]}: right ${fmt(ks[r].Ks[c].P_correct, 2)} of the time (redo-all=${fmt(ks[r].full, 2)})`}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        Fewest notes needed to fully recover (K&nbsp;★):{' '}
        {ks.filter((m) => m.K_star != null).map((m) => `${m.label} ${m.K_star}`).join(' · ')}
        {' — '}and for several models no small number is enough.
      </div>
    </div>
  )
}

function ArchBar() {
  const arch = (editing.arch as any[]).filter((a) => a.reasoning.recovery != null)
  const ARCH_LABEL: Record<string, string> = {
    attention: 'full attention (GQA)', gqa: 'full attention (GQA)',
    sliding_window: 'sliding-window', hybrid: 'hybrid attn+SSM', ssm: 'pure SSM',
  }
  return (
    <BarsH
      items={arch.map((a) => ({
        label: `${a.label} — ${ARCH_LABEL[(a.arch ?? '').toLowerCase()] ?? a.arch}`,
        value: a.reasoning.recovery,
        lo: a.reasoning.ci?.[0],
        hi: a.reasoning.ci?.[1],
        color: (a.arch ?? '').toLowerCase().includes('ssm') && !(a.arch ?? '').toLowerCase().includes('hybrid')
          ? COLORS.red
          : (a.arch ?? '').toLowerCase() === 'hybrid' ? COLORS.purple : COLORS.green,
      }))}
      domain={[0, 1.1]}
      xLabel="how well the correction note recovers (step-by-step reasoning)"
      refX={[{ x: 1, label: 'perfect' }]}
      labelWidth={290}
    />
  )
}

function WeightEditTable() {
  const w = editing.weight as any
  const rows: { key: string; label: string }[] = [
    { key: 'kv_erratum', label: 'fact + correction note (in the notes)' },
    { key: 'kv_inplace', label: 'just refresh the fact (in the notes)' },
    { key: 'rome', label: 'ROME (edit the model itself)' },
    { key: 'lora_ft', label: 'LoRA (retrain part of the model)' },
  ]
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>method</th><th>changes the decision?</th><th>time to make the change</th>
          <th>leaks into other requests</th><th>side effects elsewhere</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const m = w.methods[r.key]
          if (!m) return null
          const isKV = r.key.startsWith('kv')
          return (
            <tr key={r.key} style={isKV ? { background: '#f3f8f4' } : undefined}>
              <td style={{ fontWeight: 600 }}>{r.label}</td>
              <td>{m.efficacy_deny ? '✓' : '✗ (unless it reasons step by step)'}</td>
              <td>
                {fmtMs(m.latency_ms)}
                {m.cov_estimate_s ? ` + ${m.cov_estimate_s.toFixed(0)} s setup` : ''}
              </td>
              <td style={{ color: (m.isolation_contamination ?? 0) > 0 ? 'var(--red)' : 'var(--green)', fontWeight: 600 }}>
                {fmt(m.isolation_contamination ?? 0, 1)}
              </td>
              <td style={{ color: (m.collateral_drift ?? 0) > 0 ? 'var(--red)' : 'var(--green)', fontWeight: 600 }}>
                {fmt(m.collateral_drift ?? 0, 1)}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export function Editable() {
  return (
    <Section meta={META}>
      <P>
        As a model reads a prompt, it builds up a kind of <strong>notebook of notes</strong> — a
        saved summary of what it has read so far, so it does not have to re-read the prompt for
        every word it writes. The problem: what happens when a small fact in the prompt
        <em> changes</em> mid-conversation — an order goes from &ldquo;pending&rdquo; to
        &ldquo;shipped,&rdquo; a user&rsquo;s role changes? We&rsquo;d like to fix the notes instead
        of re-reading the whole prompt. The obvious shortcut — quietly rewrite the few notes about
        that one fact — gets ignored: the model still acts on the <em>old</em> value (we explain why
        below). There are <strong>two fixes that actually work</strong>, and neither makes the model
        re-read everything.{' '}
        <strong>(1) Redo the affected notes</strong> — reliably if you redo all the notes that came
        after the changed fact, or more cheaply but <em>less reliably</em> if you redo only the few
        most important ones (how few depends on the model).{' '}
        <strong>(2) Add a correction note</strong> — one short, clear line that the model reads as a
        fresh, trustworthy instruction. The correction note is the <strong>cheap, reliable
        default</strong>: you only add to the notes, so everything before it stays reusable (which is
        what makes the serving speedups later possible).
      </P>

      <H3>The cheap fix only works if the model thinks out loud</H3>
      <P>
        Why does the obvious shortcut fail? Because by the time you change the fact, the model has
        often <em>already written the conclusion down</em> elsewhere in its notes. Refreshing the
        one fact does not erase that conclusion, so the model keeps acting on it. This points to a
        near-free third option: refresh only the changed fact (about 1% of the work) and trust the
        model to look back at it. That works — but only when the model actually re-reads the fact
        later. A model that <strong>thinks out loud</strong>, writing out its reasoning step by step
        before answering (what researchers call &ldquo;chain-of-thought&rdquo;), does exactly that:
        the reasoning re-reads the fresh fact and recovers the right answer. Run the
        <em> identical</em> change on the <em>same</em> model with step-by-step thinking turned off,
        and it gets ignored — the model commits to the old conclusion. The deciding factor is whether
        the model reasons step by step, not how big it is (some models reason step by step by
        default; others answer directly). The details are simple, so we keep this short and tuck the
        mechanics away below.
      </P>

      <Figure
        narrow
        label="The step-by-step gap."
        title="The same cheap fix: it works when the model thinks out loud, and fails when it doesn’t"
        sub="Qwen3-8B, chance of the right decision after each fix (same model, thinking on or off); ‘redo affected notes’ = redo all notes after the changed fact"
        caption={
          <>
            Both real fixes — redo the affected notes, or add a correction note — work either way.
            The cheap &ldquo;just refresh the fact&rdquo; option is the divider: it&rsquo;s right{' '}
            <b>every</b> time when the model thinks step by step, and <b>never</b> when it
            doesn&rsquo;t — same model, both ways.
          </>
        }
      >
        <ReasoningGap />
      </Figure>

      <P>How the methods trade off cost against getting the answer right:</P>

      <Figure
        label="Cost versus getting it right."
        title="There’s no free fix, but there is a cheap one"
        sub={`Qwen3-8B, ${editing.baseline.n_tasks} decision tasks; x is a log scale`}
        caption={
          <>
            No single method wins on everything. Moving the fact to the end is cheapest but means
            rewriting the prompt; <b>fact + correction note gets it right just as reliably with no
            rewriting</b>, just one extra line; refreshing only the fact is almost free but recovers
            nothing unless the model reasons step by step; and CacheBlend, which redoes the
            most-changed notes, fails here because it follows the changed wording instead of the
            notes that already wrote down the conclusion.
          </>
        }
      >
        <Frontier />
      </Figure>

      <H3>The middle-ground fix, judged fairly</H3>
      <P>
        There&rsquo;s a tempting middle ground: redo the changed fact plus just the handful of notes
        it affects most. It works — sometimes. Even when the model reasons step by step, how many
        notes you have to redo swings wildly from model to model, because how stubbornly a model
        clings to the conclusion it already wrote down has little to do with its size:
      </P>
      <Figure
        label="Redo the fact plus its few most important notes."
        caption={
          <>
            How the chance of the right answer climbs as you redo more notes, for each model. The 8B
            model needs only about 4 extra notes, while the 4B model needs more than 64; with
            step-by-step thinking off, no small number helps at any size (it never recovers). So this
            is a real tool, but an <em>unreliable</em> one — handy when the model isn&rsquo;t stuck
            on its old conclusion, not a safe default.
          </>
        }
      >
        <KsweepHeat />
      </Figure>

      <Figure
        narrow
        label="The fix relies on the model’s ability to look back."
        caption={
          <>
            How well the correction note works across different model designs. It&rsquo;s strong on
            the common designs that can freely look back at any earlier note, weaker on a hybrid
            design, and weak on one that keeps only a rolling summary with no way to look back at a
            specific earlier note. The fix depends on the model being able to re-read a note added
            late.
          </>
        }
      >
        <ArchBar />
      </Figure>

      <H3>Why not just change the model instead?</H3>
      <P>
        A fair question: to act on a changed fact, why not edit the model itself? Two well-known
        ways to do that are ROME (a precise tweak to the model) and LoRA (a quick partial retrain).
        We compare against a careful version of ROME — first checking it on a textbook example
        (correcting where the Eiffel Tower is) so the comparison isn&rsquo;t rigged against it. All
        three approaches do change the target decision. The difference is everything else:
      </P>
      <Figure
        label="Fixing the notes vs. changing the model (Llama-3.1-8B)."
        caption={
          <>
            Changing the model is <em>global</em>: the same running model can no longer treat one
            order as &ldquo;shipped&rdquo; while another is still &ldquo;pending.&rdquo; Every order
            that&rsquo;s genuinely still pending gets wrongly flipped (the change leaks into 100% of
            other requests), and it disturbs half of an unrelated set of decisions too. The
            correction note instead lives in <em>that one conversation&rsquo;s</em> notes: it
            doesn&rsquo;t leak, it has no side effects, and it&rsquo;s 30–50&times; faster.
            Changing the model is for lasting global facts; a fact that differs from request to
            request is exactly what fixing the notes is for.
          </>
        }
      >
        <WeightEditTable />
      </Figure>

      <Aside>
        <b>Rule of thumb.</b> Default to <strong>fact + correction note</strong>: reliable, and it
        only adds to the notes, so everything before it stays reusable. If your model already reasons
        step by step and the situation is simple, just refreshing the fact is a free fast path — the
        reasoning re-reads it. Save the <strong>redo-the-few-most-important-notes</strong> option for
        models you&rsquo;ve actually tested.
      </Aside>
    </Section>
  )
}
