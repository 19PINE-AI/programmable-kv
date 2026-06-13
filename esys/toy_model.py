"""A minimal analytical model of memoization that derives the empirical findings in closed form.

We model the decision token's readout as a single attention head over the cached tokens:

    y(D) = Σ_t α_t · v_t ,     α = softmax(scores),     decision = sign(y(D))

with a 1-D signed value channel: a field value FIELD=old maps to v=+1, FIELD=new to v=-1, and the
gated decision is its sign. The empirical structure (§5: causal KV-patching, attention) is encoded
directly:
  • a FIELD token carries the value but the decision attends to it only weakly (α_F ≈ 0.1%);
  • m DOWNSTREAM "conclusion" tokens C_1..C_m *memoized* the field-conditioned value at prefill
    (v_{C_i} = +1 for FIELD=old) and carry most of the decision's attention, recency-weighted;
  • attention sinks carry value 0.

From this we read off, in closed form, why each editkv strategy behaves as observed, and we
reproduce the suffix-concentrated causal-patching curve and the dose-response. No training; the
point is that the *memoization structure alone* forces the phenomenon. Run: python esys/toy_model.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)


def build(m=40, alpha_sink=0.40, gamma=0.14):
    """Unified RECENCY-weighted attention over [field, C_1..C_m] (+ a separate sink).
    The field is the OLDEST token (position 0); the m memoized conclusion tokens follow it.
    Recency weight ∝ exp(gamma·position) ⇒ (i) α_field is tiny when m is large (the field is old)
    but grows as m shrinks (the dose-response), and (ii) recent conclusions dominate (suffix-
    concentration). Returns α over [field, C_1..C_m, sink]."""
    pos = np.arange(m + 1)                               # field at 0, C_1..C_m at 1..m
    w = np.exp(gamma * pos)
    w = w / w.sum() * (1.0 - alpha_sink)                 # field + conclusions share (1-sink)
    return np.concatenate([w, [alpha_sink]])            # [field, C_1..C_m, sink]


def readout(alpha, v_field, v_down, v_sink=0.0):
    """y(D) = α·v.  alpha = [field, C_1..C_m, sink]; v_down is the length-m conclusion vector."""
    return alpha[0] * v_field + float(np.dot(alpha[1:-1], v_down)) + alpha[-1] * v_sink


def main():
    m = 40
    alpha = build(m=m, alpha_sink=0.40, gamma=0.14)
    OLD, NEW = +1.0, -1.0                                # FIELD=old -> +1 (decision +), new -> -1
    v_down_stale = np.full(m, OLD)                       # conclusion tokens memoized OLD at prefill

    # ---- the four strategies, in closed form ----
    y_stale = readout(alpha, v_field=OLD, v_down=v_down_stale)                 # nothing changes
    y_full = readout(alpha, v_field=NEW, v_down=np.full(m, NEW))               # field + ALL conclusions -> NEW
    y_inplace = readout(alpha, v_field=NEW, v_down=v_down_stale)               # only field refreshed
    # erratum: APPEND an override token E (value NEW) with SALIENCE s -- the decision weights an
    #   explicit "[STATE UPDATE] overrides any earlier conclusion" disproportionately. Closed form:
    #   y_erratum = (y_stale + s·base·NEW) / (1 + s·base);  it flips when s·base > stale-positive mass.
    base = float(np.exp(0.14 * (m + 1)))                # the override's raw (recency) weight at the end
    base = base / (np.exp(0.14 * np.arange(m + 2)).sum())  # normalized into the distribution
    def y_err(s):
        aE = s * base
        return (y_stale + aE * NEW) / (1.0 + aE)
    s_grid = np.linspace(0, 60, 600)
    yvals = np.array([y_err(s) for s in s_grid])
    flip_idx = np.argmax(yvals < 0) if (yvals < 0).any() else -1
    s_star = float(s_grid[flip_idx]) if flip_idx > 0 else None    # salience threshold to flip
    s_use = (s_star * 1.6) if s_star else 30.0                    # a salient override above threshold
    y_erratum = y_err(s_use)

    res = {"alpha_field": float(alpha[0]), "alpha_downstream": float(alpha[1:-1].sum()),
           "alpha_sink": float(alpha[-1]), "erratum_salience_threshold_s": round(s_star, 2) if s_star else None,
           "erratum_salience_used": round(s_use, 1),
           "decision_sign": {"stale": int(np.sign(y_stale)), "full_reprefill": int(np.sign(y_full)),
                             "in_place": int(np.sign(y_inplace)), "erratum": int(np.sign(y_erratum))},
           "y": {"stale": round(y_stale, 4), "full_reprefill": round(y_full, 4),
                 "in_place": round(y_inplace, 4), "erratum": round(y_erratum, 4)}}

    # ---- suffix vs prefix causal-patching curves (reproduce Fig. fig_memoization_map) ----
    denom = (y_full - y_stale)
    fracs = np.linspace(0, 1, 21)
    suffix_rec, prefix_rec = [], []
    for fr in fracs:
        k = int(round(fr * m))
        vd = v_down_stale.copy()
        if k > 0:
            vd[m - k:] = NEW                              # patch the last k conclusions (suffix)
        suffix_rec.append((readout(alpha, OLD, vd) - y_stale) / denom)
        vd = v_down_stale.copy()
        if k > 0:
            vd[:k] = NEW                                  # patch the first k (prefix)
        prefix_rec.append((readout(alpha, OLD, vd) - y_stale) / denom)
    # field-only recovery (patch only the field value): the analytical analogue of D1's 0.009
    field_only_rec = (readout(alpha, NEW, v_down_stale) - y_stale) / denom

    # ---- dose-response: in_place recovery vs amount of memoized downstream (fewer C => later field) ----
    dose = []
    for mm in [40, 20, 10, 4, 1, 0]:
        al = build(m=mm, alpha_sink=0.40, gamma=0.14)
        ys = readout(al, OLD, np.full(mm, OLD)); yf = readout(al, NEW, np.full(mm, NEW))
        yip = readout(al, NEW, np.full(mm, OLD))
        dose.append({"m_downstream": mm, "alpha_field": round(float(al[0]), 3),
                     "in_place_recovery": round((yip - ys) / (yf - ys), 3)})

    res.update({"field_only_recovery": round(field_only_rec, 4),
                "suffix_recovery@0.1": round(suffix_rec[2], 3), "suffix_recovery@0.2": round(suffix_rec[4], 3),
                "prefix_recovery@0.5": round(prefix_rec[10], 3), "dose_response": dose})
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "toy_model.json"), "w"), indent=2)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax[0].plot(fracs, suffix_rec, "o-", color="#1f77b4", label="suffix (recent conclusions)")
    ax[0].plot(fracs, prefix_rec, "s--", color="#ff7f0e", label="prefix (early conclusions)")
    ax[0].axhline(field_only_rec, color="crimson", ls=":", label=f"field-only = {field_only_rec:.3f}")
    ax[0].set_xlabel("fraction of conclusion tokens patched to NEW"); ax[0].set_ylabel("decision recovery")
    ax[0].set_title("Analytic memoization model:\nsuffix-concentrated recovery (cf. Fig. memoization_map)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    names = ["stale", "in_place", "erratum", "full"]
    ys = [y_stale, y_inplace, y_erratum, y_full]
    cols = ["#7f7f7f" if np.sign(y) > 0 else "#2ca02c" for y in ys]
    ax[1].bar(names, ys, color=cols)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylabel("decision readout y(D)  (sign = action)")
    ax[1].set_title("in_place stays on the OLD side (stale);\nerratum & full flip to NEW")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_toy_model.png")); plt.close(fig)

    print("==== ANALYTIC TOY MODEL ====")
    print(f"  attention: field={alpha[0]:.3f}  downstream={alpha[1:-1].sum():.3f}  sink={alpha[-1]:.3f}")
    print(f"  decision readout y(D):  stale={y_stale:+.3f}  in_place={y_inplace:+.3f}  "
          f"erratum={y_erratum:+.3f}  full={y_full:+.3f}   (sign>0 = OLD action, <0 = NEW)")
    print(f"  => in_place stays OLD (FAILS); erratum & full flip to NEW (WORK)")
    print(f"  field-only recovery = {field_only_rec:.4f} (cf. empirical 0.009); "
          f"suffix@0.1={suffix_rec[2]:.2f} prefix@0.5={prefix_rec[10]:.2f}")
    print(f"  dose-response (in_place recovery vs #downstream): "
          f"{[(d['m_downstream'], d['in_place_recovery']) for d in dose]}")
    print("TOY_MODEL_DONE")


if __name__ == "__main__":
    main()
