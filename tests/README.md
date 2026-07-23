# Tests

Lightweight tests for the deepfake-detection pipeline. Each file runs standalone
(no pytest required) or under `pytest` if installed.

```bash
# run all
for f in tests/test_*.py; do python "$f"; done

# or with pytest
pytest tests/
```

- `test_grouping.py` — identity/source grouping leakage-safety invariants
- `test_forensic_features.py` — forensic descriptor shape, determinism, response
- `test_attn_head.py` — head shapes, forensic projection, missing-face NaN safety
