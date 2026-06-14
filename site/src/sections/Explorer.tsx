import { Fragment, useState } from 'react'
import { Section, P } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg, ModelPicker } from '../components/ui/Controls'
import { SegmentedPrompt } from '../components/diagrams/SegmentedPrompt'
import prompts from '../data/prompts.json'

const META = { id: 'explorer', num: '14', title: 'See the real prompts' }

/** Plain prompt box with substring highlights (first occurrence each). */
function HighlightedPrompt({ text, marks, maxHeight = 420 }: { text: string; marks: { substr: string; cls: string }[]; maxHeight?: number }) {
  let nodes: (string | JSX.Element)[] = [text]
  marks.forEach((m, mi) => {
    if (!m.substr) return
    const next: (string | JSX.Element)[] = []
    nodes.forEach((n, ni) => {
      if (typeof n !== 'string') return next.push(n)
      const i = n.indexOf(m.substr)
      if (i === -1) return next.push(n)
      next.push(n.slice(0, i))
      next.push(<span key={`${mi}-${ni}`} className={m.cls}>{m.substr}</span>)
      next.push(n.slice(i + m.substr.length))
    })
    nodes = next
  })
  return <div className="prompt-box" style={{ maxHeight }}>{nodes.map((n, i) => <Fragment key={i}>{n}</Fragment>)}</div>
}

type Treatment = 'original' | 'changed' | 'erratum' | 'hoist'

function ScenarioBrowser() {
  const scns = prompts.scenarios as any[]
  const [key, setKey] = useState('account_role')
  const [treat, setTreat] = useState<Treatment>('original')
  const s = scns.find((x) => x.key === key)!

  const fieldLineOld = `${s.label}: ${s.v_old}`
  const fieldLineNew = `${s.label}: ${s.v_new}`

  return (
    <div>
      <Controls>
        <ControlGroup label="scenario">
          <ModelPicker
            models={scns.map((x) => ({ id: x.key, label: `${x.key} (${x.cls}-conditioning)` }))}
            value={key}
            onChange={(k) => setKey(k)}
          />
        </ControlGroup>
        <ControlGroup label="prompt version">
          <Seg
            options={['original', 'changed', 'erratum', 'hoist'] as const}
            value={treat}
            onChange={setTreat}
            labels={{ original: 'original', changed: 'one fact changed', erratum: 'note added at end', hoist: 'fact moved to end' }}
          />
        </ControlGroup>
      </Controls>

      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', margin: '0 0 10px' }}>
        <code>{s.label}</code>: <span className="hl-field">{s.v_old}</span> → <span className="hl-diff">{s.v_new}</span>
        &nbsp;·&nbsp; the right answer changes from <code>{s.exp_old}</code> to <code>{s.exp_new}</code>
        {treat === 'erratum' && <> &nbsp;·&nbsp; the correction note added at the end is highlighted green</>}
        {treat === 'hoist' && <> &nbsp;·&nbsp; the changeable fact is rewritten so it now sits at the end</>}
      </div>

      {treat === 'original' ? (
        <SegmentedPrompt text={s.prompt_old} segments={s.segments} maxHeight={420} />
      ) : (
        <HighlightedPrompt
          text={treat === 'changed' ? s.prompt_new : treat === 'erratum' ? s.prompt_erratum : s.prompt_hoist}
          marks={[
            { substr: s.erratum_line && treat === 'erratum' ? s.erratum_line : '', cls: 'hl-erratum' },
            { substr: treat === 'changed' || treat === 'hoist' ? fieldLineNew : fieldLineOld, cls: 'hl-field' },
            { substr: s.gate, cls: 'hl-rule' },
          ]}
        />
      )}
    </div>
  )
}

function FieldTaxonomy() {
  const fields = prompts.fields as any[]
  const [open, setOpen] = useState<string | null>(null)
  return (
    <table className="data-table">
      <thead>
        <tr><th>fact</th><th>how much it matters</th><th>old value</th><th>change tested</th><th># rules that depend on it</th></tr>
      </thead>
      <tbody>
        {fields.map((f) => (
          <Fragment key={f.key}>
            <tr onClick={() => setOpen(open === f.key ? null : f.key)} style={{ cursor: f.n_cond ? 'pointer' : undefined }}>
              <td style={{ fontWeight: 600 }}><code>{f.label}</code></td>
              <td>
                <span style={{
                  fontWeight: 700,
                  color: f.cls === 'high' ? 'var(--red)' : f.cls === 'medium' ? 'var(--orange)' : 'var(--green)',
                }}>{f.cls}</span>
              </td>
              <td><code style={{ fontSize: 11 }}>{f.old}</code></td>
              <td><code style={{ fontSize: 11 }}>{f.semantic}</code></td>
              <td>{f.n_cond}{f.n_cond > 0 && <span style={{ color: 'var(--ink-faint)' }}> {open === f.key ? '▾' : '▸'}</span>}</td>
            </tr>
            {open === f.key && f.cond_rules.length > 0 && (
              <tr>
                <td colSpan={5} style={{ background: 'var(--bg-code)' }}>
                  <div className="mono" style={{ fontSize: 11, whiteSpace: 'pre-wrap', background: 'none' }}>
                    {f.cond_rules.join('\n')}
                  </div>
                </td>
              </tr>
            )}
          </Fragment>
        ))}
      </tbody>
    </table>
  )
}

function DissociationPair() {
  const dis = prompts.dissociation as any
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
      {dis.variants.map((v: any) => {
        const parts = v.gate.split(v.trigger)
        return (
          <div key={v.trigger}>
            <div style={{ fontFamily: 'var(--sans)', fontSize: 12, marginBottom: 6 }}>
              one word changed: <code>{v.trigger}</code> → the answer becomes: <b>{v.conclusion}</b>
            </div>
            <div className="prompt-box" style={{ maxHeight: 260, fontSize: 11 }}>
              <span className="hl-field">{dis.field_label}: {dis.field_value}</span>
              <span className="dim">  ← exactly the same in both{'\n\n'}</span>
              {parts[0]}<span className="hl-diff">{v.trigger}</span>{parts.slice(1).join(v.trigger)}
              {'\n\n'}<span className="dim">user: {dis.request}</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function SkillCard() {
  const sk = prompts.skill as any
  return (
    <div>
      <HighlightedPrompt
        text={`${sk.sys}\n\n${sk.skill}\n\n${sk.task}`}
        marks={[
          { substr: 'RULE R1: A refund may be issued ONLY if order_status is "delivered". For any other status (pending, shipped, cancelled, returned) you MUST refuse the refund and escalate to a human.', cls: 'hl-rule' },
          { substr: 'order_status = "pending"', cls: 'hl-field' },
        ]}
        maxHeight={340}
      />
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 8 }}>
        This skill was prepared on its own, then dropped into the prompt right after the system
        instructions — like pasting in a pre-written note. The right decision is <b>{sk.correct}</b>,
        and the model still gets it right after the transplant (§2), even when it reasons step by step.
      </div>
    </div>
  )
}

function RecordedOutcomes() {
  const recs = prompts.thinking as any[]
  const conds = ['oracle_new', 'stale_full', 'field_only', 'oracle_old'] as const
  const LBL: Record<string, string> = {
    oracle_new: 'fresh notes, new value (the gold standard)',
    stale_full: 'old notes reused, never updated',
    field_only: 'just the one fact patched in the notes',
    oracle_old: 'fresh notes, but the old value',
  }
  return (
    <div style={{ display: 'grid', gap: 18 }}>
      {recs.map((r) => (
        <div key={r.scenario}>
          <div style={{ fontFamily: 'var(--sans)', fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
            {r.scenario} <span style={{ color: 'var(--ink-faint)', fontWeight: 400 }}>
              · {r.seq_len} tokens · the changed fact is {(r.field_recompute_frac * 100).toFixed(1)}% of the notes
              · old notes get the right answer: {r.stale_recovers ? 'yes' : 'no'} · patching just the fact works (with step-by-step reasoning): {r.field_only_recovers ? 'yes' : 'no'}
            </span>
          </div>
          <table className="data-table">
            <thead>
              <tr><th>state of the notes</th><th>tool chosen</th><th>thinking tokens</th><th>start of the recorded answer (word for word, cut off by the test harness)</th></tr>
            </thead>
            <tbody>
              {conds.filter((c) => r[c]).map((c) => (
                <tr key={c}>
                  <td style={{ whiteSpace: 'nowrap' }}>{LBL[c]}</td>
                  <td><code>{r[c].tool}</code></td>
                  <td>{r[c].think_tokens}</td>
                  <td className="mono" style={{ fontSize: 10.5, background: 'none' }}>{r[c].answer_head}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

const TABS = ['decision scenarios', 'which facts matter', 'one word, two answers', 'transplanted skill', 'recorded outcomes'] as const

export function Explorer() {
  const [tab, setTab] = useState<(typeof TABS)[number]>('decision scenarios')
  return (
    <Section meta={META}>
      <P>
        Everything in this study runs on prompts you can read for yourself. Poke around below.
        Nothing here is faked or rebuilt for show: these are the exact prompts the model saw and
        the exact answers it gave. As you change a fact, watch how the right answer changes with
        it. Throughout, the "notes" means the model's working memory of the prompt — the running
        set of notes it builds while reading (researchers call it the KV cache).
        (For each answer we kept the tool the model picked, how many tokens it spent thinking, and
        the opening of its reply. We did not save the full reply, so we only show what was recorded.)
      </P>

      <Figure narrow>
        <div className="tabs" style={{ marginBottom: 16 }}>
          {TABS.map((t) => (
            <button key={t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>
        {tab === 'decision scenarios' && <ScenarioBrowser />}
        {tab === 'which facts matter' && <FieldTaxonomy />}
        {tab === 'one word, two answers' && <DissociationPair />}
        {tab === 'transplanted skill' && <SkillCard />}
        {tab === 'recorded outcomes' && <RecordedOutcomes />}
      </Figure>
    </Section>
  )
}
