# Queryable Memory Implementation Plan

> **For agentic workers:** Execute using superpowers:executing-plans, with a whole-branch review after verification.

**Goal:** Build and measure a compact AI association memory inspired by the SihtiBrain read/write loop.
**Architecture:** Local bounded connection writes; centered query clamps; passive value-node relaxation. Benchmark against the exact matrix equivalent, equal-state fast weights, and full-history attention.
**Tech Stack:** Python 3.10+, NumPy, Matplotlib, unittest, plain HTML/JS.
**Spec:** docs/superpowers/specs/2026-10-19-queryable-memory-design.md

## Global Constraints
- CPU only; no external datasets, pretrained weights or network requirements.
- W float32, 64x8; eta=0.5; leak=1; evaluation seeds 41,97,127,211,307.
- No tuning on evaluation episodes; 20 episodes/seed and loads 8,16,32,64,128.
- Report failed comparisons, bytes and exact matrix equivalence.

## Review Focus
- Empty/constant query must return zero scores without NaN.
- Invalid values, dimensions and nonfinite inputs must fail explicitly.
- Reads and activity reset must preserve connection memory.
- Rewrite evaluation must keep old and new truth separate.
- Demo numerical path must match Python and work offline.

### Task 1: Memory block and controls
Files: aibrain/memory.py, aibrain/baselines.py, tests/test_memory.py.
Interfaces: split(x)->(float,array); ConnectionMemory.write(x,v), read(x),
relax(x,steps=80)->(array,energy_trace), clear_activity(), erase(), state_bytes.
Baselines: HebbianMemory, DeltaMemory, HistoryAttention with write/read/state_bytes.
- [x] Write failing unittest cases for reconstruction, association recall, matched
  pooling, persistent writes, destructive erasure, energy descent, direct/iterative
  equality, byte budgets, empty cues and invalid inputs.
- [x] Run unittest and inspect missing implementation failure.
- [x] Implement block and baselines, then run full suite.
- [x] Commit working primitive.

### Task 2: Held-out experiment
Files: aibrain/benchmark.py, tests/test_benchmark.py, requirements.txt.
Interfaces: episode(seed,load,dim=64,values=8); run_benchmark(seeds,episodes,loads)
returns JSON-serializable receipts, paired metrics, rewrite and mechanism gates.
- [x] Write failing reproducibility, balanced-data, disjoint corruption, budget,
  no-evaluation-write and changed/retained-truth tests.
- [x] Run failing tests; implement exact protocol from spec.
- [x] Run suite and full benchmark; preserve per-seed receipts.
- [x] Generate docs/gate1.svg from results; commit experiment.

### Task 3: Visual demo and documentation
Files: demo/index.html, demo/memory.js, tests/test_demo.mjs,
README.md, docs/GATE1.md, .github/workflows/test.yml.
Interfaces: JS module exported for Node tests and browser; same writer/read as Python.
- [x] Write failing cross-language numerical parity test using Python fixtures.
- [x] Implement offline cue/write/query controls and connection heatmap.
- [x] Run Node parity tests and Python suite; inspect screenshot if available.
- [x] Document actual results, commands, limitations and links; commit.

### Task 4: Review and delivery
- [x] Review whole diff against spec; fix material findings with regression tests.
- [x] Run full Python and Node tests, benchmark reproduction and whitespace check.
- [x] Push feature branch and create reviewable pull request in AIBrain only.


## Completion record
Implemented in a fresh dedicated checkout and feature branch. The user authorized
building and measuring this experiment; execution proceeded inline. Whole-branch
independent review completed; two numeric input defects were fixed with failing
regressions first. See docs/VERIFICATION.md for checks and browser QA limitation.
