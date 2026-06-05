# Editable KV cache, explained from scratch

*A gentle walkthrough of what this project is, the problem it solves, and what we
found. No prior knowledge assumed. The dense technical version is
[`FINDINGS_FINAL.md`](FINDINGS_FINAL.md).*

---

## 1. The 30-second version

AI agents (think: a customer-support bot) read a long set of instructions before
every action. To go fast, the system **saves its "reading notes"** so it doesn't
re-read the same instructions every turn. The problem: if even one small value in
those instructions changes — the current time, or whether a user's account is
suspended — today's systems throw away *all* the notes and re-read everything. That's
slow and wasteful.

We asked: **can we just edit the notes in place instead of re-reading?** The answer is
**yes** — and the cheapest reliable way is to leave the old notes mostly untouched and
**staple a short "correction sticky-note" to the end** ("the account is now suspended;
this overrides anything above"). The agent reads the sticky note and does the right
thing. This recomputes ~6% instead of 100%, keeps the instructions in their natural
order, and is even *more reliable* than re-reading everything from scratch.

---

## 2. Background: what is a "KV cache" and why should I care?

When a large language model (LLM) processes text, it does it in two phases:

1. **Reading the prompt** (called *prefill*). The model goes through the whole prompt —
   system instructions, policies, the conversation so far — and builds an internal
   summary of what it has read. This summary is a big pile of numbers called the
   **KV cache** (KV = "keys and values"). Think of it as the model's **working notes**:
   for every word it read, it stored some notes about how that word relates to
   everything before it.

2. **Writing the answer** (called *decode*). The model then generates its reply one word
   at a time, each time consulting those notes.

Reading the prompt is the expensive part. For a long prompt (say 1,000–1,500 words),
prefill dominates the cost and the user-visible delay ("time to first token").

**Prefix caching** is the standard speed trick: if the next request starts with the
*exact same* text as a previous one, reuse the saved notes for that shared beginning
and only read the new part. Production systems (vLLM, SGLang) all do this.

**The catch — and the whole reason this project exists:** prefix caching only works if
the beginning matches *exactly, character for character*. Change one word early in the
prompt and every note after it is considered invalid and thrown away.

---

## 3. The problem: "mutable fields" wreck the cache

Real agent prompts contain values that **change over time** — we call these **fields**:

- the current time / date,
- a request ID or turn counter,
- the user's subscription tier, locale, region,
- the user's **account status or role** (e.g. `verified_admin` vs `suspended_user`),
- a safety mode.

These often belong *near the top* of the prompt (in a "session context" block), with the
rules that depend on them written *below*. Example rule: *"If the account is suspended,
refuse all refunds and escalate to the trust team."*

Now the agent runs turn after turn. The moment the time ticks or the account status
flips, that early value changes → prefix caching is invalidated → the system re-reads
the entire 1,500-word prompt from scratch. Every turn. This is a real, documented pain
point (e.g. Claude Code's per-turn changing header; vLLM/open-webui issues).

Today's workaround is to **move every changing value to the very end of the prompt**
("hoisting"), so the unchanging part stays cached. It works, but it forces programmers
to write prompts in an unnatural order dictated by the cache, not by what reads well or
where a value logically belongs. That tax is what we want to remove.

---

## 4. The idea: edit the notes instead of re-reading

The dream — **"editable KV"** — is: when a field changes, surgically **edit just the
affected notes** and reuse everything else, so the field can stay where it naturally
belongs *and* you keep the speed of caching.

The naive version is **"leave-stale"**: keep all the old notes, recompute the notes for
only the few words of the changed value, and hope the rest is "close enough." The risk:
the notes for the rules *below* the field were written "knowing" the old value. If the
account was `admin` when the model first read "if suspended, escalate," those notes are
subtly colored by "this user is an admin." Leave them stale and the model might keep
acting as if the user were still an admin — a potentially serious mistake.

So the real research question is: **when is it safe to leave the rest stale, and when it
isn't, what's the cheapest way to fix it without re-reading everything?**

---

## 5. Our running example

A customer-support agent for an online store. Its prompt has:
- a **policy** with a rule: *"If `account_role` is `verified_admin`, perform refunds.
  If it is `suspended_user`, refuse and call `escalate(...)` instead."*
- a **field** near the top: `account_role: verified_admin`.
- a user request: *"Please refund \$40 to my order."*

Correct behavior:
- account `verified_admin` → call `issue_refund(...)`.
- account `suspended_user` → call `escalate(...)` (do **not** refund).

The experiment: start with the prompt cached as `verified_admin`, flip the field to
`suspended_user`, edit the cache cheaply, and check whether the agent now does the
**safe** thing (`escalate`) or the **unsafe** thing (`issue_refund` anyway).

We tested on open models (Qwen2.5-7B, Qwen3-8B, Qwen3-14B), measuring (a) the *decision*
the agent makes, and (b) how much of the prompt we had to recompute (lower = cheaper).

---

## 6. What we found (the story)

**Step 1 — the free lunch is real.** Everything *before* the changed field is provably
untouched by the change (a word can't be influenced by something it was read before).
So the entire policy/preamble above the field is reusable for **free**, exactly. Good
start — but the interesting fields sit above their rules, so the rules' notes *are*
affected.

**Step 2 — first, bad news (and a wrong turn we corrected).** With a model that just
blurts out an answer (no reasoning), the cheap "leave-stale" edit **fails** for the hard
cases: the agent keeps doing the old action. To fix it you'd have to recompute almost
everything — at which point the old "hoist to the end" trick is cheaper and correct. Our
first conclusion was therefore pessimistic. *We initially over-stated this as a general
law; a follow-up showed the amount you must recompute actually depends on the field, and
for some fields a small patch suffices.*

**Step 3 — the twist: real agents think first.** Modern tool-using agents don't blurt —
they **reason out loud first** ("chain-of-thought") before acting. This matters enormously
here, because the reasoning is generated *fresh* and it **re-reads the field's current
value while it thinks.** So we re-ran with a thinking model.

**Step 4 — with thinking, the cheapest edit works.** Refresh *only* the changed field's
notes (~0.1% of the prompt), leave everything else stale, and let the model think. It
now reasons:
> *"The account role here is **suspended_user**… the rule says if suspended I must
> **escalate**… so I shouldn't process the refund."*
…and does the right thing. We confirmed this is causal: if we *don't* refresh the field,
the model reproduces the **old** decision; refreshing just that one value flips it to the
correct new one. So the live reasoning "papers over" the stale notes below.

**Step 5 — but it's not bulletproof.** On a bigger model, and when we sample the
reasoning multiple times (reasoning is a bit random), the tiny field-only edit sometimes
wobbles — occasionally cautious, occasionally stuck on the old action. And we built a
deliberately nasty case: what if the stale notes contain the agent's *own earlier
conclusion* — "I checked, the refund is allowed, I'll proceed"? Then the field-only edit
gets **fooled**: the model trusts its stale prior conclusion and refunds anyway.
(Strikingly, even fully re-reading the prompt is fooled by that planted conclusion.)

**Step 6 — the robust fix: a sticky note ("erratum").** Instead of relying on the model
to notice a quietly-changed value, **append one short, loud correction at the end** of
the prompt:
> `[STATE UPDATE] account_role has changed to suspended_user; this overrides any earlier
> value AND any earlier conclusion.`

Recompute only that ~20-word note (~6% of the prompt), leave the rest stale. Because it's
recent and explicit, the model treats it as authoritative. In our nasty "poisoned" test,
the sticky note produces the **safe** action every time — with or without thinking, and
even where fully re-reading the prompt failed. An explicit instruction beats a silent
value change.

---

## 6b. Two kinds of model — and why reasoning models are the point

There are two kinds of model you might run an agent on:

- **Instruction models** answer in one shot: they read the prompt and immediately say
  what to do, with no visible reasoning.
- **Reasoning models** (the kind powering most serious agents today) *think out loud
  first* — a paragraph of step-by-step reasoning — and then act.

This distinction is the crux. Earlier research on reusing/editing the model's notes was
done mostly with **instruction** models, and there the cheap "just fix the one value"
trick **doesn't work**: with no thinking step, nothing re-reads the corrected value, so
the model rides on its stale notes and makes the old decision. That's why prior methods
fall back to recomputing the affected notes.

**Reasoning models change this.** Because they re-read the field while thinking, the
cheap field-only fix suddenly works (in ordinary contexts). Since real agents are
reasoning models, that's where we focus. Instruction models remain the harder case — and
notably, the **sticky-note (erratum) trick works for both kinds**, because it puts the
correction in plain sight rather than relying on the model to reason its way to it.

## 7. The recipe

| Situation | Do this | Recompute cost |
|---|---|---|
| You want a safe default that always works | **Erratum**: leave cache stale, append the sticky-note correction at the end, recompute only it | **~6%** |
| Thinking model, ordinary (non-adversarial) context | **Field-only**: refresh just the changed value's notes, leave the rest stale, let it reason | **~0.1%** |
| Extra safety | **Field + erratum** (do both) | ~6% |
| (Reference) today's workaround | Move the field to the end ("hoist") | ~3.5%, but forces unnatural prompt order |
| (Baseline) what systems do now | Re-read the whole prompt | 100% |

**Bottom line:** keep the field where it naturally belongs, and either refresh just that
value (cheap, thinking models, benign context) or — for robustness — staple a short
correction note at the end. Either way you reuse the big unchanging bulk of the prompt
and recompute a few percent instead of all of it.

---

## 8. The numbers behind the story

**Does refreshing prevent the unsafe action?** We sampled the agent's decision 6× per
method (it's slightly random) and counted how often it did the **forbidden** action after
the account flipped to suspended. Lower is safer.

| how we edited the cache | chance of the UNSAFE action |
|---|---|
| leave everything stale (don't even fix the field) | 33%–100% |
| refresh just the field | **0%** |
| append the erratum sticky-note | **0%** |

So *any* real edit drives the unsafe action to zero in ordinary contexts; leaving the
cache fully stale does not.

**In the adversarial "poisoned" test** (stale notes contain a wrong prior conclusion):

| how we edited the cache | result |
|---|---|
| leave stale / refresh just the field | does the **unsafe** action (fooled) |
| **erratum sticky-note** | does the **safe** action ✓ (even non-thinking, even where full re-read fails) |

**Cost** (fraction of the prompt re-read, vs 100% for a full re-read): field-only ≈ 0.1%,
erratum ≈ 6%. The reasoning tokens are *not* an extra cost of our method — a thinking
agent produces them anyway; we only change the cheaper "reading" phase.

---

## 9. Honest limitations

- **Field-only alone is not robust.** It can be fooled by contradictory stale notes and
  wobbles on larger models. Use the erratum when correctness matters.
- **Reasoning is random**, so single runs are noisy; we report rates over multiple
  samples. Some long-reasoning runs got cut off by our token budget, so a couple of
  numbers are "at least this good" rather than exact (the *unsafe-rate* numbers are clean;
  the *correct-rate* numbers are conservative).
- Tested on a handful of models and scenarios; not yet on the largest models, on
  real-world agent benchmarks *with* reasoning (a text-parsing limitation), or on newer
  attention variants. The exact wording of the sticky-note is a knob worth tuning.

---

## 10. Mini-glossary

- **Token** — a chunk of text (roughly a word piece) the model processes one at a time.
- **Prefill / decode** — reading the prompt vs. writing the answer.
- **KV cache** — the model's saved "reading notes" for a prompt; reusing it avoids
  re-reading.
- **Prefix caching** — reuse the notes for an identical beginning of a prompt.
- **Field** — a value in the prompt that changes over time (time, status, tier, role…).
- **Stale notes / leave-stale** — keep old cached notes after a field changed instead of
  recomputing them.
- **Gating rule** — a policy that branches on a field ("if suspended, refuse").
- **Chain-of-thought (thinking)** — the model reasoning step-by-step before answering.
- **Erratum** — a short, explicit correction appended at the end of the prompt.
- **Hoisting** — the current workaround of moving changing values to the prompt's end.
