import { Section, P, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { COLORS } from '../components/charts/core'
import { fmtX, fmtPct } from '../lib/format'
import { ttft32, servingThroughput, apcErratum, apcBaseline, memTtftLo, memTtftHi } from '../lib/headline'

const META = { id: 'challenge', num: '1', title: 'The challenge' }

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
        When a model reads a prompt, it writes itself a notebook of working notes so it doesn&rsquo;t
        have to re-read the prompt over and over. Reusing those notes is what makes AI assistants
        fast. But two everyday situations break that shortcut. The first is a{' '}
        <strong>skill</strong> — a long set of instructions or tools the assistant should be able to
        load anywhere. The trouble is that the notes for a skill depend on <em>where</em> the text
        sits in the prompt, so dropping the same skill into a new spot makes the saved notes
        useless. The second is a <strong>user memory</strong> — a profile the assistant re-reads
        every turn that also <em>changes during the conversation</em>. Change even one word and every
        note written after it has to be thrown out. In both cases, today&rsquo;s systems give up and
        re-read everything from scratch, and that gets dramatically slower as the text grows.
      </P>

      <Figure narrow>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          <ChallengeCard
            tag="Challenge 1"
            title="Loading skills"
            scenario="A ready-made set of instructions or tools the assistant should be able to load once and drop in anywhere — a system prompt, a helper's prompt, a passage it just looked up."
            statusQuo="The saved notes depend on where the text sits. Move the skill to a new spot and the notes no longer fit, so the assistant re-reads it all — and the work grows much faster than the text does."
            ours="Work out the skill's notes a single time, shift them to fit the new spot, and paste them in. No re-reading, and the pasted skill still drives the right decision."
            result={`${fmtX(ttft32, 1)} faster first token at 32k tokens · same decisions as reading it all again`}
            href="#composable"
            hrefLabel="See: load a skill once (§2)"
          />
          <ChallengeCard
            tag="Challenge 2"
            title="User memory"
            scenario="A growing profile the assistant re-reads every turn — and that gets updated mid-conversation when the assistant learns something new."
            statusQuo="Put memory up front and every change forces a re-read of everything after it. Put it at the end and the assistant re-reads memory every single turn. Quietly fixing one detail in place doesn't take effect."
            ours="Treat memory like a skill that can also change: work out its notes once, shift them to fit each turn, and record updates as a short correction tacked on at the end — with nothing lost."
            result={`${fmtX(memTtftLo, 1)}–${fmtX(memTtftHi, 1)} faster first token each turn · update in place, decisions stay faithful`}
            href="#memory"
            hrefLabel="See: user memory (§4)"
          />
        </div>
      </Figure>

      <P>
        This adds up in practice. Because each correction is just tacked on at the end, the rest of
        the saved notes stay intact and reusable. Under real-world traffic that pushes the speedup to{' '}
        <strong>{fmtX(servingThroughput, 1)}</strong> when the system is fully loaded — the assistant
        can reuse its saved notes {fmtPct(apcErratum, 0)} of the time instead of{' '}
        {fmtPct(apcBaseline, 0)} (<a href="#systems">§5</a>).
      </P>

      <Aside>
        <b>How this differs from earlier work on reusing notes.</b> Reusing notes regardless of where
        text sits is not new (EPIC, CacheSlide, CacheBlend). Our two ideas sit alongside that work.
        First, we insist on getting the <em>right answer</em>: reused or edited notes have to lead the
        assistant to the <em>same decision</em>, not just run faster. Second, we add the ability to{' '}
        <em>change</em> saved notes in place — earlier systems can only paste in fixed,
        unchanging pieces. User memory is where both ideas matter at the same time.
      </Aside>
    </Section>
  )
}
