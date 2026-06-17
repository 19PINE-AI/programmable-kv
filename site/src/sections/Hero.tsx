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
          Every turn, an AI assistant re-reads the same{' '}
          <b style={{ color: 'var(--blue)' }}>skills</b> and the same big{' '}
          <b style={{ color: 'var(--orange)' }}>memory</b> of what it knows about you — and does the
          reading work all over again. As a model reads, it jots notes for itself. We let you{' '}
          <b style={{ color: 'var(--blue)' }}>reuse</b> those notes and{' '}
          <b style={{ color: 'var(--orange)' }}>amend</b> them, so the model loads things once and
          keeps using them — even as they change — without re-reading, and without changing the
          answer it would have given.
        </div>
        <div className="byline">
          <span><b>Author</b> {(constants.paper_meta as any).author} · {(constants.paper_meta as any).affiliation}</span>
          <span><b>Status</b> {constants.paper_meta.status}</span>
          <span>
            <b>Paper</b>{' '}
            <a href={(constants.paper_meta as any).arxiv} target="_blank" rel="noreferrer">
              arXiv:{(constants.paper_meta as any).arxiv_id}
            </a>
          </span>
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
          <Stat big={fmtX(ttft32, 1)} label="faster to start answering when loading a large skill" />
          <Stat big={`${fmtX(servingThroughput, 1)}`} label={`more requests served at once — the saved notes are reused ${fmtPct(apcErratum, 0)} of the time instead of ${fmtPct(apcBaseline, 0)}`} />
          <Stat big={`${fmtX(memTtftLo, 1)}–${fmtX(memTtftHi, 1)}`} label="faster to start each reply in a live assistant that remembers you" />
          <Stat big={`${agrLo.toFixed(2)}–${agrHi.toFixed(2)}`} label="how often reused notes give the same answer as redoing the work, across 13 models — reuse loses nothing" />
        </div>
        <div style={{ fontFamily: 'var(--sans)', fontSize: 11.5, color: 'var(--ink-faint)', marginBottom: 4 }}>
          Every chart on this page is drawn straight from the published results — nothing is sketched by hand.
        </div>
      </div>

      <div className="prose">
        <p className="lede">
          When a model reads a prompt, it doesn&rsquo;t just store the words. It works things out and
          writes the conclusions into a kind of scratch notebook &mdash; what engineers call the{' '}
          <strong>KV cache</strong>, the model&rsquo;s notes. Today those notes can only be reused if
          the next prompt starts with the <em>exact</em> same words. But the two things an assistant
          reuses most break that rule: a saved <strong>skill</strong> that shows up in a different
          spot each time, and a <strong>memory</strong> of what it knows about you that changes as
          you talk. So today the model throws the notes away and re-reads from scratch, over and over.
        </p>
        <p>
          This page starts with what that waste costs and what we get back (§1), then the two fixes
          &mdash; <a href="#composable">loading a skill once</a> (§2) and{' '}
          <a href="#editable">amending notes in place</a> (§3) &mdash; the{' '}
          <a href="#memory">memory feature</a> that needs both (§4), and the{' '}
          <a href="#systems">payoff for running it at scale</a> (§5). Only then do we open up{' '}
          <a href="#mechanism">why it works</a> (§6&ndash;§8). The one idea: as a model reads, it has
          already worked out its conclusions and written them down &mdash; so its notes are something
          you can read back and rewrite.
        </p>
      </div>

      <Figure
        narrow
        caption={
          <>
            The one idea, in brief. As it <b style={{ color: 'var(--orange)' }}>reads</b> the prompt,
            the model works out its conclusions and writes them onto later words (✎ notes). When it{' '}
            <b style={{ color: 'var(--blue)' }}>answers</b>, it reads back those notes &mdash; not the
            original text. That&rsquo;s what makes the notes safe to reuse and to amend; the evidence
            is in §6&ndash;§8.
          </>
        }
      >
        <NotesTeaser />
      </Figure>
    </>
  )
}
