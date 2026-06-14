import { useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Heatmap, ramp } from '../components/charts/Heatmap'
import { BarsH } from '../components/charts/BarCI'
import { COLORS } from '../components/charts/core'
import { fmt } from '../lib/format'
import reach from '../data/reach.json'

const META = { id: 'reach', num: '11', title: 'Where this works' }

interface Card {
  status: string
  color: string
  title: string
  items: string
  body: JSX.Element
}

function VariantMap() {
  const mla = (reach.mla as any[]).find((m) => m.tag === 'dscoderv2_mla')
  const mlaWeak = (reach.mla as any[]).find((m) => m.tag === 'dsv2lite_mla')
  const fix = reach.gemma_fix as any[]
  const [open, setOpen] = useState<number | null>(1)

  const cards: Card[] = [
    {
      status: 'WORKS FREE', color: COLORS.green, title: 'speed tricks that keep one note per word',
      items: 'FlashAttention · paged attention / vLLM · GQA / MQA',
      body: (
        <>These are common tricks for making models run faster. Because our idea works on the{' '}
        <em>notes themselves</em>, not on how the model reads them, it carries over with no extra
        work — and every model we tested already uses these tricks.</>
      ),
    },
    {
      status: 'SMALL ADAPTER', color: COLORS.blue, title: 'a few designs that store notes a bit differently — easy to bridge',
      items: 'MLA (DeepSeek-V2) · interleaved M-RoPE (Qwen3-VL)',
      body: (
        <>
          Some models save space by storing each note in a compressed form, where a small piece of
          the note records <em>where</em> the word sat in the text. When we move notes around, we
          only need to update that small piece — a tiny bit of bridging code. On one DeepSeek model
          this matched the from-scratch answer <b>{fmt(mla?.agreement, 2)}</b> of the time, a near-perfect{' '}
          <b>{fmt(mla?.cos, 3)}</b> match on the model&rsquo;s raw scores, across {mla?.n} decisions.
          (On a weaker version of the model it was {fmt(mlaWeak?.agreement, 2)} ({fmt(mlaWeak?.cos, 3)}) —
          we report this honestly; the paper cites the stronger result.)
          <br />
          The same idea handles image inputs: moving an image only updates its position marker — see
          the panel below.
        </>
      ),
    },
    {
      status: 'CONFIG FIX', color: COLORS.purple, title: 'models that normally keep only recent notes (Gemma) — one setting to change',
      items: 'Gemma-2 · Gemma-3',
      body: (
        <>
          By default these models throw away older notes to save memory, which leaves nothing to
          move. If you simply tell the model to <em>keep</em> all its notes (it can still choose to
          read only the recent ones), our idea works normally again: the agent that used to fail
          now matches the from-scratch answer{' '}
          <b>{fix.map((f) => fmt(f.agreement, 2)).join(' / ')}</b> of the time ({fix.map((f) => f.label).join(', ')}).
          The only thing we need is that the model keeps one note per word.
        </>
      ),
    },
    {
      status: 'PARTIAL', color: COLORS.orange, title: 'hybrid models — half notebook, half running summary',
      items: 'Falcon-H1',
      body: (
        <>These models mix two styles: one part keeps a note per word (which we can move), and one
        part keeps only a single running summary instead of separate notes. We can reuse the
        note-keeping part, but the running-summary part has to be rebuilt — so we save only some of
        the work.</>
      ),
    },
    {
      status: 'OPEN FRONTIER', color: COLORS.gray, title: 'newest 2026 designs that merge several words into one note',
      items: 'DeepSeek-V4 CSA/HCA · DeepSeek-V3.2 DSA',
      body: (
        <>To save even more memory, these brand-new designs bundle several words into a single note.
        Our idea should still apply, but it would work in chunks rather than word-by-word — we have
        worked this out on paper but not yet built it. One of them is just the compressed style
        above with an extra filter, so it can reuse the same bridging code.</>
      ),
    },
    {
      status: 'OUT OF SCOPE', color: COLORS.red, title: 'designs that keep no per-word notes at all',
      items: 'pure-recurrent (RWKV) · pure-SSM (Mamba) · diffusion LMs',
      body: (
        <>A few model designs keep only one running summary and never store a separate note per
        word. You can still fix things by editing the text you feed in, but there are no per-word
        notes to move — so this idea doesn&rsquo;t apply to them.</>
      ),
    },
  ]

  return (
    <div style={{ display: 'grid', gap: 8 }}>
      {cards.map((c, i) => (
        <div
          key={c.status}
          onClick={() => setOpen(open === i ? null : i)}
          style={{
            background: '#fff', border: '1px solid var(--rule)', borderLeft: `4px solid ${c.color}`,
            borderRadius: 6, padding: '10px 16px', cursor: 'pointer',
          }}
        >
          <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', fontFamily: 'var(--sans)' }}>
            <span style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: '0.07em', color: c.color, minWidth: 92 }}>
              {c.status}
            </span>
            <span style={{ fontSize: 13.5, fontWeight: 600 }}>{c.title}</span>
            <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--ink-faint)' }}>{open === i ? '−' : '+'}</span>
          </div>
          <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, color: 'var(--ink-faint)', marginTop: 2 }}>{c.items}</div>
          {open === i && (
            <div style={{ fontFamily: 'var(--sans)', fontSize: 13, color: 'var(--ink-soft)', lineHeight: 1.6, marginTop: 8 }}>
              {c.body}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function VisionHeat() {
  const vis = reach.vision as any[]
  const cats = ['perception', 'reasoning', 'agentic']
  return (
    <Heatmap
      rows={vis.map((v) => v.label)}
      cols={cats}
      value={(r, c) => vis[r].by_category?.[cats[c]]?.agreement ?? null}
      colorOf={(v) => (v === null ? '#f0eee6' : ramp(v, [61, 111, 180]))}
      colLabel="reusing saved image notes vs. processing the image from scratch — how often they agree, by task type (120 image questions per model)"
      rowLabelWidth={150}
      tooltip={(r, c) =>
        `${vis[r].label} · ${cats[c]}: agree ${fmt(vis[r].by_category[cats[c]].agreement, 2)} (n=${vis[r].by_category[cats[c]].n})`}
    />
  )
}

function VisionShift() {
  const sh = reach.shift as any[]
  return (
    <BarsH
      items={sh.map((s) => ({
        label: s.label,
        value: s.overall?.agreement,
        lo: s.overall?.agreement_ci?.[0],
        hi: s.overall?.agreement_ci?.[1],
        color: COLORS.blue,
      }))}
      domain={[0.5, 1.05]}
      xLabel="moving an image to a new spot (updating only its position marker) vs. from scratch — how often they agree"
      refX={[{ x: 1, label: 'identical' }]}
      labelWidth={165}
    />
  )
}

export function Reach() {
  return (
    <Section meta={META}>
      <P>
        As it reads, a model writes down a short note about each word — a kind of running notebook.
        Our whole approach edits and rearranges those notes. So the real question isn&rsquo;t which
        brand of model you have; it&rsquo;s simply: <strong>does the model keep one note per word?</strong>{' '}
        If it does, the idea works. Below we sort the recent models into six groups — click a row to
        see why.
      </P>

      <Figure
        label="Which models this works on."
        caption={
          <>
            From left to right: works for free, needs a small bridge, needs a setting changed, works
            partially, an open frontier, and out of scope. In every case that needs a bit of work,
            the only thing we touch is the small part of each note that records where the word sat —
            the notes themselves carry the rest.
          </>
        }
      >
        <VariantMap />
      </Figure>

      <H3>Images get notes too</H3>
      <P>
        Models that can see take notes about images, just like they do about words. Normally,
        re-using an image means processing it all over again — slow work. Instead, we save the
        image&rsquo;s notes once and reuse them, re-running only the text around it.
      </P>

      <Figure
        narrow
        label="Reusing saved image notes."
        caption={
          <>
            Almost no loss in quality across three kinds of image tasks on four vision-capable
            models (they agree with the from-scratch answer 95.8%–100% of the time). And moving an
            image to a new spot only takes updating its position marker — true for both ways these
            models lay out their notes.
          </>
        }
      >
        <VisionHeat />
        <div style={{ height: 18 }} />
        <VisionShift />
      </Figure>

      <Aside>
        <b>Big or small, compressed, or split into specialists.</b> Reusing notes stays faithful
        across a wide range of model sizes, on models squeezed to use less memory (a 70B model
        shrunk to 4 bits still matched at 0.986), and on models that route each word to different
        specialist parts. The single agent of §9 spans all thirteen models we tested. What matters
        is keeping one note per word — not how big the model is.
      </Aside>
    </Section>
  )
}
