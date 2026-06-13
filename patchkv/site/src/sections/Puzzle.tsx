import { useState } from 'react'
import { Section, P, Aside } from '../components/ui/Section'
import { Figure } from '../components/ui/Figure'
import { Controls, ControlGroup, Seg } from '../components/ui/Controls'
import { SegmentedPrompt } from '../components/diagrams/SegmentedPrompt'
import prompts from '../data/prompts.json'
import editing from '../data/editing.json'

const META = { id: 'puzzle', num: '1', title: 'The puzzle: a surgical edit that the model ignores' }

type Treatment = 'stale_full' | 'field_only' | 'oracle_new'

const TREATMENTS: { key: Treatment; label: string; desc: string }[] = [
  { key: 'stale_full', label: 'reuse the stale cache', desc: 'keep every cached key/value from the old prefill' },
  { key: 'field_only', label: 'refresh only the field’s KV', desc: 'recompute the 2 field tokens (~0.2% of the cache), reuse the rest' },
  { key: 'oracle_new', label: 'full recompute (oracle)', desc: 'clean prefill of the whole context with the new value' },
]

export function Puzzle() {
  const scn = prompts.scenarios.find((s) => s.key === 'account_role')!
  const rec = (editing.thinking as any[]).find((t) => t.scenario === 'account_role')!
  const [treatment, setTreatment] = useState<Treatment>('field_only')

  const out = rec[treatment]
  const correct = out.tool === scn.exp_new

  return (
    <Section meta={META}>
      <P>
        Here is the paper&rsquo;s running example, verbatim from the released harness: a
        customer-support agent with a policy document, a tool catalog, and one{' '}
        <strong>mutable field</strong> —{' '}
        <code>account_role: verified_admin</code>. A binding rule later in the prompt says a{' '}
        <code>suspended_user</code> must never get write actions; the correct next tool call
        flips from a refund lookup to <code>escalate(queue="trust", …)</code> the moment the role
        changes. Now suppose the role <em>does</em> change mid-session, and the ~1k-token prefill
        is sitting in cache.
      </P>

      <Figure
        narrow
        label="The test case."
        caption={
          <>
            The verbatim gated-decision prompt (scenario <code>account_role</code>, regenerated
            from the released harness). The <span className="hl-field">mutable field</span> and
            the <span className="hl-rule">gating rule</span> are highlighted; thirty-eight neutral
            filler rules are collapsed.
          </>
        }
      >
        <SegmentedPrompt text={scn.prompt_old} segments={scn.segments as any} maxHeight={380} />
      </Figure>

      <P>
        The cache <em>before</em> the field is provably reusable: when the field value changes,
        the keys and values of every earlier token deviate by exactly <strong>0.0</strong> — by
        construction of causal attention, those tokens never saw the field. So one might hope to
        surgically refresh only the field&rsquo;s own keys and values — two tokens, about{' '}
        <strong>0.2%</strong> of the cache — and keep everything else. Try it:
      </P>

      <Figure
        narrow
        label="Recorded behavior (Qwen3-8B, reasoning mode)."
        caption={
          <>
            Real recorded outcomes from <code>thinking_qwen3_8b_think.json</code>: the world
            changed to <code>account_role: suspended_user</code>; each treatment decides the next
            tool call. The recorded answer head is shown verbatim (the harness stores the tool
            call and thinking-token count, not full chains).
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
                stale_full: 'reuse stale cache',
                field_only: 'refresh field KV only',
                oracle_new: 'full recompute',
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
            Under explicit reasoning the in-place edit <em>can</em> recover the decision — but look
            at the cost: <b>{rec.field_only.think_tokens}</b> thinking tokens vs.{' '}
            <b>{rec.oracle_new.think_tokens}</b> for the oracle. The recomputation didn&rsquo;t
            disappear; it moved into the chain-of-thought, which re-reads the refreshed field and
            re-derives the conclusion. <b>Without reasoning, the same edit recovers nothing</b> —
            P(correct) ={' '}
            {(editing.baseline.methods.find((m: any) => m.method === 'in_place')!.P_correct as number).toFixed(2)}{' '}
            across the editing benchmark (§6). The model simply acts on the old value, as if the
            edit never happened.
          </div>
        )}
      </Figure>

      <Aside>
        <b>The puzzle.</b> The field&rsquo;s new value is sitting right there in the cache — every
        key and every value of those two tokens is fresh — and the prefix before it never depended
        on the field at all. Yet the decision reverts to the <em>old</em> value. Something else in
        the cache must be carrying the old conclusion. The next section finds it, causally.
      </Aside>
    </Section>
  )
}
