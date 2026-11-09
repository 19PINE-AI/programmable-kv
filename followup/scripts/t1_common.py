"""Shared instance set for T1/T7/T1b: the FULL polarity 2x2 of the repo's circuit harness.

For every scenario (3) x order id (4) x field value (2) we build a conclusion-flip pair with the
FIELD held byte-identical: SAFE prompt has trigger == field, UNSAFE prompt has trigger == other
value. That is every field-controlled SAFE/UNSAFE pair the repo's polarity task defines (24).
circuit_common.build_pair only uses field = values[0]; this generalizes it to values[1] too.
"""
import os, sys
REPO = os.environ.get("PKV_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "esys"))
sys.path.insert(0, os.path.join(REPO, "e1"))
import circuit_common as cc  # noqa
from align import align_pair  # noqa
from mechd_common import POL, build_pol  # noqa

TRAIN_OIDS = ["A4471", "B8820", "C1093"]
HELD_OID = "D5567"
INSTANCES = [(scn, oid, fi) for scn in cc.SCNS for oid in cc.OIDS for fi in (0, 1)]


def build_pair_f(tok, scn, oid, fi, thinking=False):
    s = POL[scn]
    fld = s["values"][fi]
    other = s["values"][1 - fi]
    t_safe = build_pol(tok, scn, oid, fld, fld, thinking, True)
    t_unsafe = build_pol(tok, scn, oid, fld, other, thinking, True)
    al = align_pair(tok, t_safe, t_unsafe)
    safe_ids, unsafe_ids = al["old_ids"], al["new_ids"]
    a, b = al["field_span"]
    L = safe_ids.shape[1]
    dpos = L - 1
    last = int(safe_ids[0, dpos])
    toi = {"safe": cc.ftok(tok, cc.ACT_TOK[s["safe"]]), "unsafe": cc.ftok(tok, cc.ACT_TOK[s["unsafe"]])}
    return dict(safe_ids=safe_ids, unsafe_ids=unsafe_ids, trig_span=(a, b), L=L,
                dpos=dpos, last=last, toi=toi, scn=scn, oid=oid, fi=fi)
