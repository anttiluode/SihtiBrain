# SihtiBrain first experiment implementation plan

**Goal:** Make the proposed dendrite / soma / spike analogy falsifiable using a transparent, reproducible toy circuit.

**Architecture:** `sihtibrain.py` owns the symmetric tree and numerical solver. `demo.py` runs fixed inputs, computes metrics, and draws the trace. `tests/test_model.py` checks analytical invariants and the four gates; `README.md` interprets the result against published neurobiology.

**Tech Stack:** Python 3, NumPy, Matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-10-19-sihti-brain-design.md`

## Tasks

1. Write failing tests for passive energy descent, the exact branch exchange symmetry, and locally observable difference at equal soma. Implement the minimal circuit solver, run tests, commit.
2. Write a failing test for input-matched branch coincidence and threshold event. Implement the branch term and threshold readout, run tests, commit.
3. Implement deterministic CLI visualization and JSON output. Verify artifact generation and document the model, limits, primary papers, and follow-up controls. Run whole suite, inspect outputs, commit.

## Review focus

Check that energy is asserted only for the frozen, passive phase; that plots do not imply the cell discovers objects; that the compared stimuli have identical magnitude and depth; that branch current signs are explicit; and that the phenomenological coincidence term is not called an NMDA simulation.
