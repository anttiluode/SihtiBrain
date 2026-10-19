# Verification — 19 October 2026

- Python: `python -m unittest discover -s tests -v` — 16 tests pass.
- JavaScript: `node tests/test_demo.mjs` — Python/JS numeric parity, reconstruction,
  activity reset, connection erasure, invalid inputs and large finite cues pass.
- Two complete repeat benchmark runs reproduced all per-seed/per-episode recall
  and rewrite accuracies, both summary tables, storage budgets and mechanism gates.
  Numerical input fixes do not change the reported scores.
- An independent read-only code/science review reproduced the full experiment.
  It found no target leakage, contaminated rewrite truth or unsupported baseline win.
- Review fixes: absolute probability tolerance plus normalization now preserves
  bounded conductance; scaling before query centering/normalization prevents
  overflow from large finite inputs. Both failures were reproduced in tests before
  implementation and pass after the fixes, in Python and JavaScript.
- Generated SVG whitespace was normalized; the four-panel plot was rendered to
  a PNG and visually inspected for readable labels and layout.
- The offline demo's model code and static local dependencies were checked.
  Browser rendering and UI interactions were **not visually verified**: no browser
  executable was available, and the Chromium download returned an invalid archive.
  No browser-test success is claimed.

CI runs Python tests, JS parity and a small benchmark on Python 3.10 / Node 20.
Local results use Python 3.12 / NumPy 2.3.5. CI status is separate from local tests.
