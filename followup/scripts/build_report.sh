#!/bin/bash
# Assemble ../REPORT.md = narrative (report_text.md, sections marked @@NAME@@) + tables generated
# by make_tables.py from the records in ../results (or $FOLLOWUP_OUT). CPU only.
cd "$(dirname "$0")"
python3 - <<"PY"
import subprocess
txt = open("report_text.md").read()
def tab(*a):
    return subprocess.run(["python3", "make_tables.py", *a], capture_output=True, text=True).stdout
for name, args in [("CAUSAL_LLAMA", ["causal", "llama31_8b"]), ("CAUSAL_QWEN", ["causal", "qwen3_8b"]), ("LABELS", ["labels"]),
                   ("GAO", ["gao"]), ("T7", ["t7"]), ("T1B", ["t1b"]), ("OV", ["ov"]), ("SPLIT", ["split"])]:
    txt = txt.replace(f"@@{name}@@", tab(*args))
open("../REPORT.md", "w").write(txt)
print("REPORT.md written", len(txt))
PY
