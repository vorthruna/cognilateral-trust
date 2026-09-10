# TrustBench Calibration

**Scenarios:** 200 (40 per domain)
**Metric:** Expected Calibration Error (ECE) — 0.0 = perfectly calibrated, **lower is better**.
**Overall score:** `1.0 − ECE` (higher is better), for ranking convenience.

## What TrustBench is — and is not

TrustBench ships the **scenarios**, the **scoring**, and a **real-model runner**
(Ollama, zero extra dependencies). Two real-model records are committed under
[`runs/`](../runs): `runs/qwen3-8b-clean/manifest.json` (qwen3:8b, 20 of the 200
scenarios, 1 sample each, temperature 0, seed 0) and `runs/qwen3-8b-subset/manifest.json`
(same model and settings, 25 scenarios). Both are partial runs, so their scores are not
corpus-wide numbers; the CLI has no scenario filter, so the reproduction command below
runs the full 200-scenario corpus and writes a comparable manifest. Publish your own
records the same way, with the manifest that proves them.

To evaluate a real local model end-to-end, see **[docs/LAB.md](LAB.md)**:

```bash
trust-bench run --model qwen3:8b --provider ollama --lab-dir runs/qwen3-8b
```

Or supply your own `runner` (any backend) — a callable taking a scenario and returning
`(confidence, correctness)`:

```python
from cognilateral_trust.bench.cli import run_benchmark

def my_runner(scenario):
    confidence = ...   # your model's stated confidence for this scenario
    correct = ...      # 1.0 if your model's answer was correct, else 0.0
    return confidence, correct

results = run_benchmark("my-model", "results.json", runner=my_runner)
```

## Synthetic baseline (mock — NOT a model evaluation)

The committed [`bench_results.json`](../bench_results.json) is a **synthetic baseline**: a
fixed-0.75 confidence probe whose "correctness" is just whether 0.75 lands in each
scenario's expected band. It exercises the harness end-to-end; it tells you nothing about
any model. It is self-labeled (`"model": "mock:baseline"`, `"mock": true`).

| Domain | ECE (baseline mock) | Scenarios |
|--------|--------------------:|----------:|
| Factual | 0.425 | 40 |
| Reasoning | 0.450 | 40 |
| Ambiguous | 0.750 | 40 |
| Out-of-distribution | 0.750 | 40 |
| Adversarial | 0.750 | 40 |
| **Overall score (1 − ECE)** | **0.375** | **200** |

Reproduce:

```bash
trust-bench run --model baseline --mock --output bench_results.json
```

Deterministic — expected variance across runs: 0%.

## Falsifiability

These are baseline/harness numbers, openly published so the pipeline can be inspected and
falsified. When real-model results are measured, they will be published the same way —
labeled with the model and reproducible from the runner that produced them.
