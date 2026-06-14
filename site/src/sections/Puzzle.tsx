import { useState } from 'react'
import { Section, P, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg } from '../components/ui/Controls'
import { SegmentedPrompt } from '../components/diagrams/SegmentedPrompt'
import prompts from '../data/prompts.json'
import editing from '../data/editing.json'

const META = { id: 'puzzle', num: '6', title: 'Why it works: the edit the model ignores' }

type Treatment = 'stale_full' | 'field_only' | 'oracle_new'

const TREATMENTS: { key: Treatment; label: string; desc: string }[] = [
  { key: 'stale_full', label: 'reuse the stale cache', desc: 'keep all the old notes the model made while reading the original prompt' },
  { key: 'field_only', label: 'refresh only the field’s notes', desc: 'rewrite just the two words of the changed field (about 0.2% of the notes), keep the rest' },
  { key: 'oracle_new', label: 'reread everything', desc: 'have the model reread the whole prompt with the new value — the ideal answer to compare against' },
]

export function Puzzle() {
  const scn = prompts.scenarios.find((s) => s.key === 'account_role')!
  const rec = (editing.thinking as any[]).find((t) => t.scenario === 'account_role')!
  const [treatment, setTreatment] = useState<Treatment>('field_only')

  const out = rec[treatment]
  const correct = out.tool === scn.exp_new

  return (
    <Section meta={META}>
      <Aside>
        <b>Part II — why it works.</b> First, a quick idea. As a model reads a prompt, it writes
        itself a private notebook of working notes — one note per word. (Researchers call this
        notebook the &ldquo;KV cache.&rdquo;) Saving the notebook lets the model skip rereading the
        same text later. Everything in Part I rested on one claim: you can move or amend those notes
        and the model still behaves correctly. The rest of the page earns that claim — starting with
        the obvious shortcut that <em>fails</em>.
      </Aside>
      <P>
        Here is the example we&rsquo;ll use throughout, taken straight from our test code. A
        customer-support assistant has a policy document, a list of tools it can call, and one{' '}
        <strong>changeable detail</strong> — the user&rsquo;s role,{' '}
        <code>account_role: verified_admin</code>. A rule further down says a{' '}
        <code>suspended_user</code> must never be allowed to make changes. So the moment the role
        switches, the right next action flips: instead of looking up a refund, the assistant should
        call <code>escalate(queue="trust", …)</code>. Now imagine the role really does change
        partway through the session — and the model&rsquo;s notebook of notes (about a thousand
        words&rsquo; worth) is already saved.
      </P>

      <Figure
        narrow
        label="The test case."
        caption={
          <>
            The actual prompt we test (scenario <code>account_role</code>, regenerated from our test
            code). The <span className="hl-field">changeable detail</span> and the{' '}
            <span className="hl-rule">rule that depends on it</span> are highlighted; thirty-eight
            other harmless rules are collapsed to save space.
          </>
        }
      >
        <SegmentedPrompt text={scn.prompt_old} segments={scn.segments as any} maxHeight={380} />
      </Figure>

      <P>
        The notes taken <em>before</em> the changeable detail are guaranteed safe to keep. A model
        reads strictly left to right, so any word written before the detail couldn&rsquo;t have seen
        it — when the detail changes, those earlier notes change by exactly <strong>0.0</strong>. So
        here&rsquo;s the tempting shortcut: just rewrite the notes for the two words of the detail
        itself — about <strong>0.2%</strong> of the notebook — and keep everything else. Try it:
      </P>

      <Figure
        narrow
        label="What actually happened (Qwen3-8B, thinking mode)."
        caption={
          <>
            Real results from our recorded runs (<code>thinking_qwen3_8b_think.json</code>): the
            role changed to <code>account_role: suspended_user</code>, and each treatment decides
            the next action. The start of the model&rsquo;s own answer is shown word for word (we
            saved the action it chose and how much it thought, not its full reasoning).
          </>
        }
      >
        <Controls>
          <ControlGroup label="cache treatment">
            <Seg
              options={['stale_full', 'field_only', 'oracle_new'] as const}
              value={treatment}
              onChange={setTreatment}
              labels={{
                stale_full: 'reuse old notes',
                field_only: 'refresh only the field',
                oracle_new: 'reread everything',
              }}
            />
          </ControlGroup>
        </Controls>

        <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-faint)', marginBottom: 12 }}>
          {TREATMENTS.find((t) => t.key === treatment)!.desc}
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr auto',
            gap: 16,
            alignItems: 'center',
            background: '#fff',
            border: `2px solid ${correct ? 'var(--green)' : 'var(--red)'}`,
            borderRadius: 8,
            padding: '14px 18px',
          }}
        >
          <div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 600 }}>
              tool_call: {out.tool}(…)
            </div>
            <div className="mono" style={{ fontSize: 11, background: 'none', color: 'var(--ink-faint)', marginTop: 6, whiteSpace: 'pre-wrap' }}>
              {out.answer_head}…
            </div>
          </div>
          <div style={{ textAlign: 'right', fontFamily: 'var(--sans)', fontSize: 12.5 }}>
            <div style={{ fontWeight: 700, fontSize: 15, color: correct ? 'var(--green)' : 'var(--red)' }}>
              {correct ? '✓ correct' : '✗ acts on the OLD value'}
            </div>
            <div style={{ color: 'var(--ink-faint)', marginTop: 4 }}>
              {out.think_tokens} thinking tokens
            </div>
          </div>
        </div>

        {treatment === 'field_only' && (
          <div style={{ fontFamily: 'var(--sans)', fontSize: 12.5, color: 'var(--ink-soft)', marginTop: 12, lineHeight: 1.6 }}>
            When the model is allowed to think out loud, the patched notebook <em>can</em> reach
            the right answer — but look at the price: <b>{rec.field_only.think_tokens}</b> words of
            thinking, versus <b>{rec.oracle_new.think_tokens}</b> when it simply rereads. The work we
            tried to save didn&rsquo;t vanish; it just moved into the thinking, where the model
            notices the refreshed detail and works out the conclusion again. <b>And when it
            isn&rsquo;t allowed to think out loud, the same patch recovers nothing</b> — it lands the
            right answer only{' '}
            {(editing.baseline.methods.find((m: any) => m.method === 'in_place')!.P_correct as number).toFixed(2)}{' '}
            of the time across our test set (§3). The model just acts on the old value, as if nothing
            had changed.
          </div>
        )}
      </Figure>

      <Aside>
        <b>The puzzle.</b> The new value is right there in the notebook — the notes for those two
        words are completely up to date — and nothing written before them ever depended on the old
        value. Yet the model still decides as if the value had never changed. Something else in the
        notebook must be quietly carrying the old conclusion. The next section tracks down exactly
        what.
      </Aside>
    </Section>
  )
}
