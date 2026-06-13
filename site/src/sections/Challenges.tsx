import { Section, P, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { COLORS } from '../components/charts/core'
import { fmtX, fmtPct } from '../lib/format'
import { ttft32, servingThroughput, apcErratum, apcBaseline, memTtftLo, memTtftHi } from '../lib/headline'

const META = { id: 'challenge', num: '1', title: 'The challenge: reusing skills and user memory' }

function ChallengeCard({
  tag,
  title,
  scenario,
  statusQuo,
  ours,
  result,
  href,
  hrefLabel,
}: {
  tag: string
  title: string
  scenario: string
  statusQuo: string
  ours: string
  result: string
  href: string
  hrefLabel: string
}) {
  return (
    <div style={{ background: '#fff', border: '1px solid var(--rule)', borderRadius: 8, padding: '18px 20px', flex: '1 1 320px', minWidth: 300 }}>
      <div style={{ fontFamily: 'var(--sans)', fontSize: 11, fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: COLORS.orange }}>{tag}</div>
      <div style={{ fontFamily: 'var(--serif)', fontSize: 19, fontWeight: 600, margin: '4px 0 8px' }}>{title}</div>
      <p style={{ fontFamily: 'var(--sans)', fontSize: 13, color: 'var(--ink-soft)', lineHeight: 1.55, margin: '0 0 12px' }}>{scenario}</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 10px', fontFamily: 'var(--sans)', fontSize: 12.5, lineHeight: 1.5 }}>
        <div style={{ fontWeight: 700, color: COLORS.red }}>Today</div>
        <div style={{ color: 'var(--ink-soft)' }}>{statusQuo}</div>
        <div style={{ fontWeight: 700, color: COLORS.green }}>Ours</div>
        <div style={{ color: 'var(--ink-soft)' }}>{ours}</div>
      </div>
      <div style={{ marginTop: 12, padding: '8px 12px', background: 'var(--bg-figure)', borderRadius: 6, fontFamily: 'var(--sans)', fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>
        {result}
      </div>
      <div style={{ marginTop: 10, fontFamily: 'var(--sans)', fontSize: 12.5 }}>
        <a href={href}>{hrefLabel} →</a>
      </div>
    </div>
  )
}

export function Challenges() {
  return (
    <Section meta={META}>
      <P>
        Two reuse patterns dominate agent serving, and both defeat ordinary prefix caching. A{' '}
        <strong>skill</strong> — a long policy or tool specification — is reused across many
        contexts, but it lands at a <em>different position</em> each time, so the cached keys no
        longer match. A <strong>user-memory</strong> document is reused every turn, but it{' '}
        <em>mutates mid-session</em>, and a single changed token invalidates the whole downstream
        cache. In both cases today&rsquo;s systems fall back to a full, quadratic re-prefill.
      </P>

      <Figure narrow>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          <ChallengeCard
            tag="Challenge 1"
            title="Loading skills"
            scenario="A precompiled policy / tool-spec the agent should load once and reuse anywhere — system prompt, sub-agent prompt, retrieved passage."
            statusQuo="Position-dependent KV: drop the skill at a new offset and the cache misses. Full reprefill is O(L²) in the skill length."
            ours="Precompile the skill once, RoPE-reposition its keys to the target offset, splice it in — O(L), no recompute, and the spliced skill still governs the decision."
            result={`${fmtX(ttft32, 1)} faster first token at 32k tokens · decision-identical to full recompute`}
            href="#composable"
            hrefLabel="See: load a skill once (§2)"
          />
          <ChallengeCard
            tag="Challenge 2"
            title="User memory"
            scenario="A large, dynamically-summarized profile the assistant re-reads every turn — and updates mid-session when a tool writes to it."
            statusQuo="Front placement reprefills everything after memory on every change; end placement re-attends memory every turn; a surgical in-place edit is silently ignored."
            ours="Treat memory as a skill that is also edited: compose it once, reposition each turn, and apply changes with an append-only erratum — losslessly."
            result={`${fmtX(memTtftLo, 1)}–${fmtX(memTtftHi, 1)} faster per-turn first token · mutate in place, decision-faithful`}
            href="#memory"
            hrefLabel="See: user memory (§4)"
          />
        </div>
      </Figure>

      <P>
        The serving consequence is large: because the edit is append-only, the static prefix stays
        cache-aligned, so under real online load the throughput advantage grows to{' '}
        <strong>{fmtX(servingThroughput, 1)}</strong> at saturation ({fmtPct(apcErratum, 0)} vs{' '}
        {fmtPct(apcBaseline, 0)} prefix-cache hit-rate; <a href="#systems">§5</a>).
      </P>

      <Aside>
        <b>What&rsquo;s new here vs. prior KV-reuse.</b> Position-independent caching already exists
        (EPIC, CacheSlide, CacheBlend). Our two contributions are orthogonal to that machinery:
        (1) a <b>decision-governance</b> lens — the reused or edited cache must still make the{' '}
        <em>right</em> tool decision, not merely run fast; and (2) the <b>editing axis</b> — you
        can <em>mutate</em> cached state in place, which prior reuse systems (they splice only{' '}
        <em>static</em> blocks) do not support. Memory is the case where both matter at once.
      </Aside>
    </Section>
  )
}
