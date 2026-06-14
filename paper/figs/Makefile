# Regenerate every paper figure from the raw result records.
#
#   make            # rebuild all figures into this directory (paper/figs/)
#   make memory     # only the user-memory figures (E1-E5, LoCoMo)
#
# Data sources: ../../results/*.json (main + appendix + circuit + horizon figures)
#               ../../mem/results/*.jsonl (user-memory figures, via mem/make_figs.py)
# Every PDF here is the output of one of these scripts; do not hand-edit the PDFs.

PYTHON ?= python3
MEM    := ../../mem

.PHONY: all main appendix circuit horizon memory
all: main appendix circuit horizon memory

main:
	$(PYTHON) make_figures.py

appendix:
	$(PYTHON) make_appendix_figures.py

circuit:
	$(PYTHON) make_circuit_figure.py

horizon:
	$(PYTHON) make_horizon_figure.py

# E1-E5 and LoCoMo render from $(MEM)/results into this directory.
memory:
	FIGS_OUT=$(CURDIR) $(PYTHON) $(MEM)/make_figs.py
