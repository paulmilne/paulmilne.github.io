# Scaling-law study (Scaling Laws, Part 1)

Character-level GPTs on text8, trained on CPU with PyTorch.

- `train.py` — trains one model, logs held-out loss (nats/char) along the run, writes a JSON.
- `plan.py` → `jobs.json`, `joblist.txt` — first 22 runs (budgets 1e12–3e13 FLOPs).
- `jobs2.json`, `joblist2.txt` — 13 further runs (smaller shapes and a 1e14 budget).
- `runall.sh`, `runall2.sh` — run the job lists two at a time.
- `python3 analyse.py out.json` (reads `runs/C*.json`) — iso-FLOP parabolas, frontier exponent, Chinchilla-form surface fit (Huber, L-BFGS), hold-out prediction of the largest budget. Runs with fewer than 1,000 optimiser steps are excluded from fits.

Data: `data/text8.u8`, text8 stored as one byte per character, values 0–26. Not included.
