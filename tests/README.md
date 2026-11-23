# Tests

These are self-contained checks that stub out the LLM layer, so they run without
a configured model endpoint.

| File | What it covers |
| --- | --- |
| `test_jev_decider.py` | `jev27b_decider` negative/positive denoising and decision logic |
| `test_scheduler_unit.py` | Report scheduler queue, dependency ordering, progress callbacks |
| `test_scheduler_fail.py` | Knowledge-chunk permanent-failure fallback and overall timeout marking |
| `test_report_concurrency.py` | End-to-end concurrent report generation over N patients |

Run them from the repository root so the top-level modules are importable:

```bash
python tests/test_jev_decider.py
python tests/test_scheduler_unit.py
python tests/test_scheduler_fail.py
python tests/test_report_concurrency.py --patients 16 --concurrency 16
```

`test_report_concurrency.py` expects patient subgraphs produced by a previous
pipeline run; it is an integration check rather than a pure unit test.
