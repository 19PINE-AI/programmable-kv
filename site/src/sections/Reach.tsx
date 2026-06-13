import { useState } from 'react'
import { Section, P, H3, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Heatmap, ramp } from '../components/charts/Heatmap'
import { BarsH } from '../components/charts/BarCI'
import { COLORS } from '../components/charts/core'
import { fmt } from '../lib/format'
import reach from '../data/reach.json'

const META = { id: 'reach', num: '11', title: 'Reach: where the substrate holds' }

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
      status: 'FREE', color: COLORS.green, title: 'throughput optimizations that keep per-token KV',
      items: 'FlashAttention · paged attention / vLLM · GQA / MQA',
      body: (
        <>The operations act on the cache <em>representation</em>, not the attention kernel — these
        transfer with zero work, and every model in the study already uses them.</>
      ),
    },
    {
      status: 'ADAPTER', color: COLORS.blue, title: 'representation changes — small adapters, implemented & validated',
      items: 'MLA (DeepSeek-V2) · interleaved M-RoPE (Qwen3-VL)',
      body: (
        <>
          <b>MLA</b> caches a position-free latent plus a small decoupled-RoPE sub-vector{' '}
          <code>k_pe</code>; the adapter re-rotates <em>only that sub-vector</em>. On
          DeepSeek-Coder-V2-Lite: composed-vs-full agreement <b>{fmt(mla?.agreement, 2)}</b>, logit
          cosine <b>{fmt(mla?.cos, 3)}</b> over {mla?.n} decisions. (On the weaker V2-Lite-Chat
          checkpoint, agreement is {fmt(mlaWeak?.agreement, 2)} at cosine {fmt(mlaWeak?.cos, 3)} —
          reported honestly; the paper cites the Coder result.)
          <br />
          <b>Interleaved M-RoPE</b>: moving an image re-rotates only the temporal axis — see the
          multimodal panel below.
        </>
      ),
    },
    {
      status: 'FIXED', color: COLORS.purple, title: 'sliding-window attention (Gemma) — a cache-layout bug, fixed',
      items: 'Gemma-2 · Gemma-3',
      body: (
        <>
          The default cache <em>truncates</em> sliding-window layers to the window, destroying the
          per-token KV a splice needs beyond it. Keeping the full per-token KV and letting the
          attention <em>mask</em> enforce the window restores uniform edit/splice semantics: the
          previously-failing unified agent runs at agreement{' '}
          <b>{fix.map((f) => fmt(f.agreement, 2)).join(' / ')}</b> ({fix.map((f) => f.label).join(', ')}).
          The substrate requirement is per-token KV — not full attention.
        </>
      ),
    },
    {
      status: 'PARTIAL', color: COLORS.orange, title: 'hybrid attention + SSM',
      items: 'Falcon-H1',
      body: (
        <>The attention KV transplants, but the per-layer Mamba scan-state is recurrent, not
        per-token — a correct transplant must re-scan the Mamba path and saves only the attention
        fraction.</>
      ),
    },
    {
      status: 'OPEN', color: COLORS.gray, title: 'the 2026 sparse / compressed-attention frontier',
      items: 'DeepSeek-V4 CSA/HCA · DeepSeek-V3.2 DSA',
      body: (
        <>Sequence-dimension KV compression merges tokens into fewer-than-token entries, so edit
        and splice become <em>block</em>-granular — analyzed but not implemented. DSA is MLA plus
        top-k selection and inherits the MLA adapter directly.</>
      ),
    },
    {
      status: 'OUT OF SCOPE', color: COLORS.red, title: 'no per-token attention KV at all',
      items: 'pure-recurrent (RWKV) · pure-SSM (Mamba) · diffusion LMs',
      body: (
        <>The prompt-level erratum still applies as plain text, but there is no per-token cache to
        edit or splice — §3&rsquo;s architecture bar shows the erratum&rsquo;s recovery decaying
        exactly along this axis.</>
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
      colLabel="spliced image-KV vs. full re-encode — decision agreement by task category (120 VQA tasks per model)"
      rowLabelWidth={150}
      tooltip={(r, c) =>
        `${vis[r].label} · ${cats[c]}: agreement ${fmt(vis[r].by_category[cats[c]].agreement, 2)} (n=${vis[r].by_category[cats[c]].n})`}
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
      xLabel="position-shifted image (temporal-axis re-rotation only) vs. full — agreement"
      refX={[{ x: 1, label: 'identical' }]}
      labelWidth={165}
    />
  )
}

export function Reach() {
  return (
    <Section meta={META}>
      <P>
        Where exactly do these operations hold? The answer is a representational condition, not an
        architectural family: <strong>anything that keeps one KV entry per token per layer</strong>.
        The paper maps the 2025–26 attention landscape against that condition — click a row:
      </P>

      <Figure
        label="The attention-variant map."
        caption={
          <>
            Free / adapter / fixed / partial / open / out-of-scope. The two implemented adapters
            (MLA&rsquo;s decoupled <code>k_pe</code> re-rotation, M-RoPE&rsquo;s temporal-axis
            re-rotation) and the sliding-window cache fix each touch only the position-dependent
            slice of the representation — the substrate does the rest.
          </>
        }
      >
        <VariantMap />
      </Figure>

      <H3>Images take notes too</H3>
      <P>
        In a vision-language agent, an image costs the vision tower <em>plus</em> prefilling
        &gt;1k soft-tokens through the LM on every reuse. Image notes are position-portable just
        like text notes: cache the image&rsquo;s LM-side KV once, splice it, re-run only the text.
      </P>

      <Figure
        narrow
        label="Image-KV transplant."
        caption={
          <>
            Near-lossless across perception, reasoning, and agentic VQA on four vision-language
            models (agreement 0.958–1.0 overall) — and moving an image to a different trajectory
            position needs only the temporal M-RoPE axis re-rotated, in both the sectioned
            (Qwen2.5-VL) and interleaved (Qwen3-VL) layouts.
          </>
        }
      >
        <VisionHeat />
        <div style={{ height: 18 }} />
        <VisionShift />
      </Figure>

      <Aside>
        <b>Scale, quantization, MoE.</b> The transplant is faithful from 0.6B to 32B, on FP8
        checkpoints, on a 30B-A3B Mixture-of-Experts, and on a 4-bit 70B (logit cosine 0.986) —
        the unified agent of §9 spans all thirteen. Per-token KV, not model size, is the
        load-bearing assumption.
      </Aside>
    </Section>
  )
}
