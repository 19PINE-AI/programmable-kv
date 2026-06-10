"""Statistical-rigor toolkit for the memory-KV experiments.

All inference respects the nesting decision <- trajectory <- persona:
  * cluster_bootstrap_*  : resample at the PERSONA (cluster) level, 1e4 by default.
  * mcnemar_exact        : paired binary contrast (two methods on the same items).
  * tost_proportions     : equivalence of paired proportions (CI-inclusion within +-delta).
  * gee_logit            : GEE logistic regression with cluster-robust SE (cluster=persona).
  * bh_fdr / holm        : multiple-comparison correction.
  * cohens_h, risk_diff  : effect sizes.

Conventions: a "record" is a dict with at least keys for the cluster id (default "persona")
and the binary/continuous outcome(s). Functions take lists of records.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Sequence, Callable, Optional
import numpy as np

RNG = np.random.default_rng(20260608)


# ----------------------------- effect sizes -----------------------------
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    def phi(p):
        p = min(max(p, 0.0), 1.0)
        return 2.0 * math.asin(math.sqrt(p))
    return phi(p1) - phi(p2)


def risk_diff(p1: float, p2: float) -> float:
    return p1 - p2


def odds_ratio(a, b, c, d):
    """OR from a 2x2 (with Haldane-Anscombe 0.5 correction)."""
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a * d) / (b * c)


# ----------------------------- cluster bootstrap -----------------------------
def _by_cluster(records, cluster_key):
    groups = defaultdict(list)
    for r in records:
        groups[r[cluster_key]].append(r)
    return list(groups.values())


def cluster_bootstrap_stat(records, stat_fn: Callable, cluster_key="persona",
                           n_boot=10000, ci=0.95, seed=None):
    """Bootstrap a scalar statistic by resampling clusters with replacement.

    stat_fn: list[record] -> float. Returns (point, lo, hi, se).
    """
    rng = np.random.default_rng(seed) if seed is not None else RNG
    clusters = _by_cluster(records, cluster_key)
    nC = len(clusters)
    point = stat_fn(records)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nC, nC)
        sample = [r for j in idx for r in clusters[j]]
        boots[i] = stat_fn(sample)
    a = (1 - ci) / 2
    lo, hi = np.nanpercentile(boots, [100 * a, 100 * (1 - a)])
    return float(point), float(lo), float(hi), float(np.nanstd(boots))


def cluster_bootstrap_diff(records, stat_fn_a: Callable, stat_fn_b: Callable,
                           cluster_key="persona", n_boot=10000, ci=0.95, seed=None):
    """Bootstrap a PAIRED difference stat_a - stat_b (same resampled clusters for both).

    Returns dict with point, lo, hi, se, and p_two_sided (bootstrap, H0: diff=0).
    """
    rng = np.random.default_rng(seed) if seed is not None else RNG
    clusters = _by_cluster(records, cluster_key)
    nC = len(clusters)
    point = stat_fn_a(records) - stat_fn_b(records)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nC, nC)
        sample = [r for j in idx for r in clusters[j]]
        boots[i] = stat_fn_a(sample) - stat_fn_b(sample)
    a = (1 - ci) / 2
    lo, hi = np.nanpercentile(boots, [100 * a, 100 * (1 - a)])
    # two-sided bootstrap p: 2 * min(frac<=0, frac>=0), centered at observed
    centered = boots - point
    p = 2.0 * min(np.mean(centered >= point), np.mean(centered <= point))
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                se=float(np.nanstd(boots)), p=float(min(1.0, p)))


# ----------------------------- paired binary -----------------------------
def mcnemar_exact(b: int, c: int):
    """Exact McNemar test on discordant counts b (a>b wins) and c.

    b = #items where A correct & B wrong; c = #items A wrong & B correct.
    Exact binomial two-sided p. Returns dict(p, b, c, n_discordant).
    """
    n = b + c
    if n == 0:
        return dict(p=1.0, b=b, c=c, n_discordant=0)
    k = min(b, c)
    # two-sided exact binomial against p=0.5
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * tail)
    return dict(p=float(p), b=int(b), c=int(c), n_discordant=int(n))


def paired_binary_counts(records, key_a, key_b):
    """Discordant counts for two boolean fields across paired records."""
    b = sum(1 for r in records if r[key_a] and not r[key_b])
    c = sum(1 for r in records if (not r[key_a]) and r[key_b])
    return b, c


# ----------------------------- equivalence (TOST) -----------------------------
def tost_proportions_paired(records, key_a, key_b, delta, cluster_key="persona",
                            n_boot=10000, alpha=0.05, seed=None):
    """Equivalence of two paired proportions within +-delta via cluster-bootstrap CI.

    Declares equivalence iff the (1-2*alpha) CI of (P(A)-P(B)) lies inside [-delta, delta]
    (the CI-inclusion form of TOST). Returns dict with diff, ci, delta, equivalent, p_tost.
    """
    pa = lambda rs: float(np.mean([r[key_a] for r in rs]))
    pb = lambda rs: float(np.mean([r[key_b] for r in rs]))
    # TOST uses a (1-2alpha) two-sided CI
    d = cluster_bootstrap_diff(records, pa, pb, cluster_key, n_boot, ci=1 - 2 * alpha, seed=seed)
    equivalent = (d["lo"] >= -delta) and (d["hi"] <= delta)
    # approximate TOST p: from each one-sided test using the bootstrap SE (normal approx)
    se = max(d["se"], 1e-9)
    from scipy.stats import norm
    p_lower = norm.cdf((d["point"] - (-delta)) / se)   # H0: diff <= -delta
    p_upper = norm.sf((d["point"] - delta) / se)        # H0: diff >= +delta
    p_tost = max(1 - p_lower, 1 - p_upper)  # both one-sided must reject; report max
    # cleaner: p_tost = max of the two one-sided p-values
    p1 = norm.sf((d["point"] + delta) / se)   # test diff > -delta  -> small p good
    p2 = norm.cdf((d["point"] - delta) / se)  # test diff < +delta  -> small p good
    p_tost = max(p1, p2)
    return dict(diff=d["point"], lo=d["lo"], hi=d["hi"], delta=float(delta),
                equivalent=bool(equivalent), p_tost=float(p_tost), se=se)


def tost_mean_paired(values_a, values_b, delta, alpha=0.05):
    """TOST for paired continuous values (e.g., per-item logit cosine vs 1.0)."""
    from scipy.stats import t
    a = np.asarray(values_a, float); b = np.asarray(values_b, float)
    d = a - b
    n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n) if n > 1 else 1e-9
    tl = (m - (-delta)) / se; tu = (m - delta) / se
    p_lower = t.sf(tl, n - 1)   # H0: mean <= -delta
    p_upper = t.cdf(tu, n - 1)  # H0: mean >= +delta
    p_tost = max(p_lower, p_upper)
    crit = t.ppf(1 - alpha, n - 1)
    ci = (m - crit * se, m + crit * se)  # (1-2alpha) CI
    return dict(diff=float(m), lo=float(ci[0]), hi=float(ci[1]), delta=float(delta),
                equivalent=bool(ci[0] >= -delta and ci[1] <= delta), p_tost=float(p_tost))


# ----------------------------- GEE logistic -----------------------------
def gee_logit(records, outcome: str, formula_rhs: str, cluster_key="persona"):
    """GEE logistic regression with exchangeable correlation, cluster-robust SEs.

    records: list of dicts. outcome: name of 0/1 field. formula_rhs: patsy RHS, e.g.
        "C(placement) * n_facts + np.log(L_mem) + C(reasoning)"
    Returns dict: params, pvalues, conf_int (cluster-robust), n, n_clusters.
    """
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    df = pd.DataFrame(records)
    df[outcome] = df[outcome].astype(int)
    fam = sm.families.Binomial()
    cov = sm.cov_struct.Exchangeable()
    model = smf.gee(f"{outcome} ~ {formula_rhs}", groups=cluster_key, data=df,
                    family=fam, cov_struct=cov)
    res = model.fit()
    ci = res.conf_int()
    return dict(params={k: float(v) for k, v in res.params.items()},
                pvalues={k: float(v) for k, v in res.pvalues.items()},
                ci_lo={k: float(ci.loc[k, 0]) for k in res.params.index},
                ci_hi={k: float(ci.loc[k, 1]) for k in res.params.index},
                n=int(len(df)), n_clusters=int(df[cluster_key].nunique()),
                summary=str(res.summary()))


# ----------------------------- multiplicity -----------------------------
def bh_fdr(pvals: Sequence[float], q=0.05):
    """Benjamini-Hochberg. Returns (rejected_bool_list, adjusted_pvals)."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n); out[order] = adj
    rej = out <= q
    return rej.tolist(), out.tolist()


def holm(pvals: Sequence[float], alpha=0.05):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    adj = np.maximum.accumulate(ranked * (n - np.arange(n)))
    adj = np.clip(adj, 0, 1)
    out = np.empty(n); out[order] = adj
    return (out <= alpha).tolist(), out.tolist()


# ----------------------------- power -----------------------------
def n_for_equivalence(delta, sigma_d, power=0.90, alpha=0.05):
    """Approx paired-difference sample size for an equivalence test."""
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha); z_b = norm.ppf(power)
    return math.ceil(((z_a + z_b) ** 2) * (sigma_d ** 2) / (delta ** 2))


def proportion_ci_wilson(k, n, ci=0.95):
    if n == 0:
        return (0.0, 0.0, 1.0)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - ci) / 2); p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, center - half), min(1.0, center + half))


if __name__ == "__main__":
    # self-test
    print("n for equivalence (delta=0.03, sigma=sqrt(0.05)):",
          n_for_equivalence(0.03, math.sqrt(0.05)))
    recs = [dict(persona=i % 10, a=RNG.random() < 0.9, b=RNG.random() < 0.9) for i in range(500)]
    print("bootstrap P(a):", cluster_bootstrap_stat(recs, lambda rs: np.mean([r["a"] for r in rs])))
    b, c = paired_binary_counts(recs, "a", "b")
    print("mcnemar:", mcnemar_exact(b, c))
    print("tost:", tost_proportions_paired(recs, "a", "b", 0.05))
    print("bh:", bh_fdr([0.001, 0.02, 0.3, 0.04]))
    print("STATS_SELFTEST_OK")
