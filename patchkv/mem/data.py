"""Controlled persona/memory data generator for the memory-KV experiments.

A persona has a Markdown user-memory of M settings (realistic account/preference facts).
A *gated decision* is governed by `n_relevant` of those settings: the agent may proceed
only if ALL relevant settings are "enabled". This gives:
  * clean integration-depth control (n_relevant = how many facts must be AND-ed),
  * a flippable gold label (toggle one relevant setting -> decision flips) for editing,
  * a negative control (toggle an IRRELEVANT setting -> decision must NOT change),
  * memory-length control (M pads the doc with decision-irrelevant settings),
  * balanced labels (half "yes", half "no") so a constant-answer prior cannot win.

Token lengths are measured by the caller's tokenizer; we expose target M presets and the
caller picks the M that lands near a desired L_mem.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional

# realistic setting catalog: (category, attribute, human note). enabled/disabled is assigned per persona.
CATALOG = [
    ("notifications", "marketing_emails", "promotional email campaigns"),
    ("notifications", "push_alerts", "mobile push notifications"),
    ("notifications", "sms_updates", "text-message status updates"),
    ("privacy", "data_sharing", "share usage data with partners"),
    ("privacy", "personalized_ads", "ad personalization from activity"),
    ("privacy", "location_history", "retain precise location history"),
    ("privacy", "third_party_export", "export records to third parties"),
    ("security", "two_factor", "two-factor authentication requirement"),
    ("security", "new_device_login", "auto-approve logins from new devices"),
    ("security", "biometric_unlock", "biometric unlock on this account"),
    ("billing", "auto_renew", "automatic subscription renewal"),
    ("billing", "overdraft_purchases", "allow purchases that overdraw balance"),
    ("billing", "saved_cards", "store card details for one-click pay"),
    ("content", "mature_content", "show age-restricted content"),
    ("content", "external_links", "open links to external sites"),
    ("content", "beta_features", "enroll in experimental beta features"),
    ("sharing", "public_profile", "make profile publicly visible"),
    ("sharing", "activity_status", "broadcast online/active status"),
    ("sharing", "contact_sync", "sync device contacts to the service"),
    ("automation", "auto_reply", "send automated replies when away"),
    ("automation", "smart_suggestions", "AI smart-suggestion features"),
    ("automation", "background_sync", "sync data in the background"),
    ("comms", "newsletter", "weekly product newsletter"),
    ("comms", "survey_invites", "invitations to user surveys"),
    ("accessibility", "high_contrast", "high-contrast display mode"),
    ("accessibility", "screen_reader", "screen-reader optimizations"),
    ("data", "cloud_backup", "encrypted cloud backups"),
    ("data", "cross_device", "cross-device session handoff"),
    ("data", "analytics_optin", "anonymous analytics opt-in"),
    ("integrations", "calendar_access", "calendar integration access"),
    ("integrations", "email_access", "email inbox integration access"),
    ("integrations", "drive_access", "cloud-drive file access"),
]

# named actions that a gated decision authorizes (purely cosmetic flavor, paired with relevant settings)
ACTIONS = [
    "send a promotional offer to the user",
    "share the user's record with a partner service",
    "auto-approve a login from an unrecognized device",
    "charge the user's saved card for renewal",
    "publish an update to the user's public profile",
    "sync the user's contacts to the cloud",
    "enroll the user in a new beta feature",
    "export the user's data to a third-party tool",
]


@dataclass
class Persona:
    pid: int
    settings: List[dict]              # [{idx, category, attr, note, enabled(bool)}]
    relevant_idxs: List[int]          # the gating settings (len == n_relevant)
    flip_idx: int                     # which relevant setting starts "disabled" (the gold violator) or is the toggle target
    action: str
    gold_yes: bool                    # True if all relevant enabled (gold answer == "yes")
    irrelevant_idx: int               # a non-gating setting (negative control)

    def memory_markdown(self) -> str:
        lines = ["# USER MEMORY (account settings & preferences)\n"]
        for s in self.settings:
            state = "enabled" if s["enabled"] else "disabled"
            lines.append(f"- [{s['category']}] {s['attr']}: {state}  ({s['note']})")
        lines.append("\n# END USER MEMORY\n")
        return "\n".join(lines)

    def relevant_names(self) -> List[str]:
        return [self.settings[i]["attr"] for i in self.relevant_idxs]

    def decision_query(self, reasoning: bool) -> str:
        names = ", ".join(self.relevant_names())
        rule = (f"TASK: The assistant wants to {self.action}.\n"
                f"RULE: Proceed ONLY IF ALL of these user settings are enabled: {names}.\n"
                f"Look up EACH of these settings in the USER MEMORY and check its value.\n")
        if reasoning:
            rule += ("Think step by step, checking each setting's value, then give your final answer.\n"
                     "After reasoning, end with a line 'FINAL: yes' or 'FINAL: no'.\n")
        else:
            rule += "Answer with exactly one word — yes (proceed) or no (do not proceed).\nAnswer:"
        return rule

    def with_toggle(self, idx: int, enabled: bool) -> "Persona":
        """Return a copy with setting `idx` set to enabled/disabled (recomputes gold)."""
        import copy
        p = copy.deepcopy(self)
        p.settings[idx]["enabled"] = enabled
        p.gold_yes = all(p.settings[i]["enabled"] for i in p.relevant_idxs)
        return p


def make_persona(seed: int, n_total: int, n_relevant: int, gold_yes: bool) -> Persona:
    """Generate a persona with n_total settings (padded/truncated catalog, repeated with
    distinct attr names if n_total > catalog) and n_relevant gating settings.

    gold_yes=True  -> all relevant enabled.
    gold_yes=False -> exactly one relevant disabled (the flip target / violator).
    """
    rng = random.Random(seed)
    # build a settings pool of size n_total by tiling the catalog with suffixes
    pool = []
    i = 0
    while len(pool) < n_total:
        c, a, note = CATALOG[i % len(CATALOG)]
        suffix = "" if i < len(CATALOG) else f"_{i // len(CATALOG)}"
        pool.append(dict(idx=len(pool), category=c, attr=a + suffix, note=note, enabled=True))
        i += 1
    rng.shuffle(pool)
    for k, s in enumerate(pool):
        s["idx"] = k
    # default: random ~60% enabled for irrelevant ones (realistic mix), but relevant handled below
    for s in pool:
        s["enabled"] = rng.random() < 0.6
    # choose relevant gating settings (spread across the doc)
    relevant = rng.sample(range(n_total), n_relevant)
    for i in relevant:
        pool[i]["enabled"] = True            # start all relevant enabled
    if gold_yes:
        flip_idx = relevant[0]               # toggle target for the editing experiment
    else:
        flip_idx = relevant[0]
        pool[flip_idx]["enabled"] = False    # exactly one violator -> gold "no"
    # negative-control irrelevant setting (not in relevant set)
    irr_candidates = [i for i in range(n_total) if i not in relevant]
    irrelevant_idx = rng.choice(irr_candidates)
    action = rng.choice(ACTIONS)
    gold = all(pool[i]["enabled"] for i in relevant)
    return Persona(pid=seed, settings=pool, relevant_idxs=relevant, flip_idx=flip_idx,
                   action=action, gold_yes=gold, irrelevant_idx=irrelevant_idx)


def make_dataset(n_personas: int, n_total: int, n_relevant: int, seed0: int = 0) -> List[Persona]:
    """Balanced dataset: half gold_yes, half gold_no."""
    out = []
    for k in range(n_personas):
        out.append(make_persona(seed0 + k, n_total, n_relevant, gold_yes=(k % 2 == 0)))
    return out


def filler_trajectory(n_turns: int, seed: int = 0) -> str:
    """Decision-irrelevant past conversation, to make placement matter."""
    rng = random.Random(seed)
    topics = ["the weather this week", "a recipe for dinner", "weekend plans",
              "a movie recommendation", "how to stay productive", "a travel idea",
              "a book worth reading", "tips for better sleep", "a fitness routine",
              "music for focus"]
    msgs = []
    for t in range(n_turns):
        topic = rng.choice(topics)
        msgs.append(f"User: Can you chat about {topic}?")
        msgs.append(f"Assistant: Sure — here are a few friendly thoughts about {topic}, "
                    f"keeping things general and light without touching your account settings.")
    return "\n".join(msgs)


if __name__ == "__main__":
    p = make_persona(0, 24, 4, gold_yes=True)
    print(p.memory_markdown()[:600])
    print("relevant:", p.relevant_names(), "gold_yes:", p.gold_yes)
    print(p.decision_query(False))
    print("--- toggled flip to disabled ---")
    p2 = p.with_toggle(p.flip_idx, False)
    print("gold_yes after toggle:", p2.gold_yes)
    ds = make_dataset(10, 24, 4)
    print("dataset gold balance:", sum(x.gold_yes for x in ds), "/", len(ds))
