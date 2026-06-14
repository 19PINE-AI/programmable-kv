import { Section, P } from '../components/ui/Section'
import constants from '../data/constants.json'

const META = { id: 'boundaries', num: '15', title: "What it can't do (yet)" }

const LIMITS = [
  {
    t: 'Only works on the usual kind of model',
    d: 'Our approach needs a model that keeps its notes as a per-word "KV cache" — the running notebook a model fills in as it reads. Some newer designs (RWKV, Mamba, and diffusion language models) store their state differently and are out of scope. Mixed designs are only partly covered, because part of their memory cannot be moved.',
  },
  {
    t: 'The near-free quick edit is not reliable',
    d: 'Editing just one fact and a few of the model’s nearby notes is cheap, but how well it sticks depends on the model and the situation: a small tweak is enough for one model (about 4 notes at 8B), while another needs far more (over 64 at 4B). We treat it as a tool you measure case by case, not a default you can trust blindly.',
  },
  {
    t: 'Simple lookups are a partial exception',
    d: 'When the task is just to look up a stored value, the model’s note is partly a literal copy of that value. So editing the fact alone still changes the answer somewhat (recovery 0.25–0.63) instead of having almost no effect.',
  },
  {
    t: 'A few rough edges remain',
    d: 'If a single transplanted piece is itself larger than Gemma’s attention window, quality dips (logit cosine ≈0.89) — though real reusable skills are far smaller. And a few small models ship older code that needs a minor patch to work with our method.',
  },
  {
    t: 'The newest compressed models: analyzed, not built',
    d: 'Some 2026 models squeeze their notes into blocks rather than per-word entries (DeepSeek-V4-class designs). With those, the smallest thing you can edit or splice is a whole block, not a single word. Editing the memory behind images is similarly still open. We analyze these cases but have not implemented them.',
  },
  {
    t: 'Some tests use made-up rules, partly addressed',
    d: 'Several of our experiments use synthetic (made-up) policies so we can measure things cleanly. We balance this with real material: an actual retail-support benchmark (τ²-bench), real images, and real recorded conversations (LoCoMo). Testing on a wider range of real workloads is still to come.',
  },
  {
    t: 'Credit where it is due',
    d: 'The underlying caching machinery (reusing notes regardless of position, recomputing at boundaries, and re-anchoring positions) is prior work — Prompt Cache, CacheBlend, EPIC, CacheSlide, MPIC. This editable-and-composable-cache direction grew directly out of EPIC and CacheSlide and out of discussions with Junhao Hu (first author of EPIC; see the acknowledgement below). What we claim as new is the explanation of how it works, the unified view, the decision-governance angle, and the adapters for different attention designs — not the reuse machinery itself.',
  },
]

export function Boundaries() {
  return (
    <Section meta={META}>
      <div className="prose">
        <div style={{ display: 'grid', gap: 10 }}>
          {LIMITS.map((l) => (
            <div key={l.t} style={{ background: '#fff', border: '1px solid var(--rule)', borderRadius: 6, padding: '12px 16px' }}>
              <div style={{ fontFamily: 'var(--sans)', fontSize: 13.5, fontWeight: 600, marginBottom: 3 }}>{l.t}</div>
              <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', lineHeight: 1.6 }}>{l.d}</div>
            </div>
          ))}
        </div>
      </div>

      <P>
        <strong>The big idea.</strong> When you edit one fact in the model’s notes and the answer
        does not change, it is not because the notes are flimsy. It is because the model already did
        the thinking earlier: as it first reads the input, it writes a conclusion into its notes,
        and the final answer just reads that conclusion back. So the model’s notebook is a record
        you can read and write — a record of what the model has already figured out. We hope that
        way of seeing it proves useful well beyond editing and combining notes.
      </P>

      <div className="colophon">
        <p style={{ fontWeight: 600, marginBottom: 8 }}>Acknowledgements</p>
        <p>
          We thank Junhao Hu, first author of EPIC (Efficient Position-Independent Caching) and a
          co-author of CacheSlide, for discussions on editable KV cache that inspired this research
          project.
        </p>
        <p style={{ fontWeight: 600, marginBottom: 8 }}>Colophon</p>
        <p>
          This page is the interactive companion to <em>{constants.paper_meta.title}</em> by{' '}
          {(constants.paper_meta as any).author} ({(constants.paper_meta as any).affiliation}),{' '}
          {constants.paper_meta.status.toLowerCase()} Code and result records:{' '}
          <a href={(constants.paper_meta as any).github} target="_blank" rel="noreferrer">
            github.com/19PINE-AI/programmable-kv
          </a>
          . Every number you see here is read straight from the released results and run logs, and
          the example prompts are regenerated by the released code, so they come out the same every
          time. The few values that appear only in the paper’s text are flagged inline with a{' '}
          <span className="paper-const">⊙ from paper text</span> badge. Recorded model outputs are
          shown exactly as they were saved (tool call, count of thinking tokens, and the start of
          the answer) — never made up after the fact.
        </p>
        <div className="citation-block">{`@article{li2026programmablekv,
  title  = {Models Take Notes at Prefill:
            KV Cache Can Be Editable and Composable},
  author = {Li, Bojie},
  note   = {Preprint. Under review.},
  year   = {2026},
  url    = {https://github.com/19PINE-AI/programmable-kv}
}`}</div>
      </div>
    </Section>
  )
}
