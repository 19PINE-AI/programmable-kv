import { NotesTeaser } from '../components/diagrams/NotesTeaser'
import { Figure } from '../components/ui/Figure'
import constants from '../data/constants.json'

export function Hero() {
  return (
    <>
      <header className="hero">
        <div className="kicker">Interactive paper companion</div>
        <h1>Models Take Notes at Prefill</h1>
        <div className="subtitle">
          The KV cache is not a frozen byproduct of prefill — it is a notebook of memoized
          conclusions. That makes it <b style={{ color: 'var(--orange)' }}>editable</b> and{' '}
          <b style={{ color: 'var(--blue)' }}>composable</b>.
        </div>
        <div className="byline">
          <span><b>Authors</b> Anonymous · under review</span>
          <span><b>Status</b> {constants.paper_meta.status}</span>
          <span><b>This page</b> every figure is driven by the released result records</span>
        </div>
      </header>

      <Figure
        narrow
        caption={
          <>
            At <b style={{ color: 'var(--orange)' }}>prefill</b>, the model computes the
            field-conditioned conclusion and writes it onto downstream aggregator/delimiter tokens
            (✎ notes). At <b style={{ color: 'var(--blue)' }}>decode</b>, the decision reads those
            notes — not the field. Schematic; the causal evidence is in §2–§5.
          </>
        }
      >
        <NotesTeaser />
      </Figure>

      <div className="prose">
        <p className="lede">
          Modern LLM agents re-read long, mostly-static instructions on every turn — a system
          policy, a tool specification, retrieved documents. Key/value caching makes this
          affordable, but only across an <em>exact</em> shared prefix: the moment one token changes
          inside the reused region — a timestamp, a user id, an order&rsquo;s status — the keys and
          values of every later token are invalidated.
        </p>
        <p>
          This page is an interactive companion to the paper. It walks through the mechanism the
          paper discovers — <strong>attention-mediated memoized inference</strong> — with the
          actual prompts, causal-patching results, and circuit-level measurements behind each
          claim, then shows how one mechanism yields two capabilities: editing a cached context
          with a one-line <em>erratum</em>, and composing precompiled <em>skills</em> into new
          contexts with no recompute. Everything you can hover, drag, or toggle below is rendered
          from the released result records — nothing is re-drawn by hand.
        </p>
      </div>
    </>
  )
}
