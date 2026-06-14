import { NotesTeaser } from '../components/diagrams/NotesTeaser'
import { Figure } from '../components/ui/Figure'
import constants from '../data/constants.json'
import { fmtX, fmtPct } from '../lib/format'
import { ttft32, servingThroughput, apcErratum, apcBaseline, memTtftLo, memTtftHi, agrLo, agrHi } from '../lib/headline'

function Stat({ big, label }: { big: string; label: string }) {
  return (
    <div style={{ flex: '1 1 150px', minWidth: 150 }}>
      <div style={{ fontFamily: 'var(--sans)', fontSize: 26, fontWeight: 700, color: 'var(--ink)', lineHeight: 1.1 }}>{big}</div>
      <div style={{ fontFamily: 'var(--sans)', fontSize: 12, color: 'var(--ink-faint)', marginTop: 4, lineHeight: 1.4 }}>{label}</div>
    </div>
  )
}

export function Hero() {
  return (
    <>
      <header className="hero">
        <div className="kicker">Interactive paper companion</div>
        <h1>Programmable KV Cache</h1>
        <div className="subtitle">
          LLM agents re-read a library of <b style={{ color: 'var(--blue)' }}>skills</b> and a big{' '}
          <b style={{ color: 'var(--orange)' }}>user-memory</b> document every turn. We make the KV
          cache <b style={{ color: 'var(--blue)' }}>composable</b> and{' '}
          <b style={{ color: 'var(--orange)' }}>editable</b>, so you load them once and reuse
          them — even as they change — without recomputing, and without changing the decision.
        </div>
        <div className="byline">
          <span><b>Author</b> {(constants.paper_meta as any).author} · {(constants.paper_meta as any).affiliation}</span>
          <span><b>Status</b> {constants.paper_meta.status}</span>
          <span>
            <b>Code</b>{' '}
            <a href={(constants.paper_meta as any).github} target="_blank" rel="noreferrer">
              github.com/19PINE-AI/programmable-kv
            </a>
          </span>
        </div>
      </header>

      <div className="prose">
        <div
          style={{
            display: 'flex',
            gap: 28,
            flexWrap: 'wrap',
            background: 'var(--bg-figure)',
            border: '1px solid var(--rule)',
            borderRadius: 8,
            padding: '20px 24px',
            margin: '28px 0 8px',
          }}
        >
          <Stat big={fmtX(ttft32, 1)} label="faster first token loading a 32k-token skill (O(L) vs O(L²))" />
          <Stat big={`${fmtX(servingThroughput, 1)}`} label={`higher serving throughput at saturation — ${fmtPct(apcErratum, 0)} vs ${fmtPct(apcBaseline, 0)} prefix-cache hit-rate`} />
          <Stat big={`${fmtX(memTtftLo, 1)}–${fmtX(memTtftHi, 1)}`} label="faster per-turn first token on a live user-memory agent" />
          <Stat big={`${agrLo.toFixed(2)}–${agrHi.toFixed(2)}`} label="decision agreement with full recompute, across 13 models — the cache is reused losslessly" />
        </div>
        <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, color: 'var(--ink-faint)', marginBottom: 4 }}>
          Every figure on this page is rendered from the released result records — nothing is drawn by hand.
        </div>
      </div>

      <div className="prose">
        <p className="lede">
          Key/value caching makes prefill reuse affordable, but only across an <em>exact</em> shared
          prefix. The two things an agent reuses most — a precompiled <strong>skill</strong> that
          lands at different positions in different contexts, and a <strong>user-memory</strong>{' '}
          document that mutates mid-session — both break that assumption, so today they are
          re-prefilled from scratch over and over.
        </p>
        <p>
          This page leads with what that costs and what we recover (§1), then the two capabilities
          that fix it — <a href="#composable">loading a skill once</a> (§2) and{' '}
          <a href="#editable">mutating in place</a> (§3) — the <a href="#memory">user-memory
          application</a> that needs both (§4), and the <a href="#systems">serving payoff</a> (§5).
          Only then do we open up <a href="#mechanism">why it works</a> (§6–§8): the one idea is that
          a transformer, at prefill, already computes its conclusions and writes them down — the
          cache is a notebook you can read and rewrite.
        </p>
      </div>

      <Figure
        narrow
        caption={
          <>
            The one idea, in brief. At <b style={{ color: 'var(--orange)' }}>prefill</b> the model
            writes field-conditioned conclusions onto downstream tokens (✎ notes); at{' '}
            <b style={{ color: 'var(--blue)' }}>decode</b> the decision reads those notes, not the
            field. This is what makes the cache safely reusable and editable — the causal evidence
            is in §6–§8.
          </>
        }
      >
        <NotesTeaser />
      </Figure>
    </>
  )
}
