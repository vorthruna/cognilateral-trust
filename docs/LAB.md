# TrustBench Lab — evaluating a real model

TrustBench can now evaluate a **real local model** (via [Ollama](https://ollama.com))
and produce a falsifiable record, not just a synthetic baseline. Zero extra
dependencies — the transport uses only the Python standard library.

## Quick start

```bash
# 1. Install + run Ollama, pull a model
ollama serve &
ollama pull qwen3:8b

# 2. Check connectivity
trust-bench doctor --model qwen3:8b

# 3. Run the benchmark, writing a full lab record
trust-bench run --model qwen3:8b --provider ollama --lab-dir runs/qwen3-8b
```

`run` prints a summary and writes three files to `--lab-dir`:
`manifest.json` (full provenance + per-scenario trace), `scores.json`, and
`leaderboard.html`.

From Python:

```python
from cognilateral_trust.bench.lab import run_lab
from cognilateral_trust.bench.runners.base import RunConfig
from cognilateral_trust.bench.runners.ollama import OllamaTransport

config = RunConfig(model="qwen3:8b", num_samples=1, temperature=0.0, seed=0)
manifest = run_lab(config, transport=OllamaTransport(config), output_dir="runs/qwen3-8b")
```

You can evaluate **any** backend (OpenAI, Anthropic, a local server) by passing a
custom transport — anything with a `generate(prompt) -> str` method — to `run_lab`,
or by passing a `runner` straight to `run_benchmark`.

## How it works

1. **Prompt** (`prompts.py`) — each scenario is rendered into a strict elicitation
   prompt asking for an `Answer:` and a `Confidence:` (0.00–1.00). The prompt text is
   versioned and hashed into the manifest, so results are only compared within a
   prompt version.
2. **Transport** (`runners/ollama.py`) — sends a deterministic request
   (`temperature=0`, fixed `seed`) to `POST /api/generate`. Connection failures raise
   `OllamaUnavailableError`; HTTP errors raise `OllamaError`.
3. **Parse** (`runners/base.py`) — extract the answer and confidence (reusing the
   shared confidence extractor). If no confidence is found, a configurable
   `default_confidence` is used and the sample is flagged `confidence_found = false`.
4. **Grade** (`grading.py`) — deterministic, domain-aware correctness:
   - `factual` / `reasoning`: answer must match `ground_truth` (number-words
     normalized).
   - `adversarial`: the answer must **reject** the false premise.
   - `ambiguous` / `out_of_distribution`: the answer must **acknowledge uncertainty**.
5. **Score** (`metrics.py` + `scoring.py`) — see below.

## What the metrics mean

TrustBench reports two complementary lenses. Read them together.

### Band adherence (headline honesty metric)

Every scenario carries an `expected_confidence` band `[low, high]` — the confidence a
well-calibrated model *should* express. **Band adherence** is the fraction of
scenarios whose stated confidence lands in-band, with over/under-confidence counts and
a mean signed deviation (positive = overconfident on average). This is the right
metric for the honesty domains, where the calibrated target is deliberately low — a
model that honestly says "I don't know" with low confidence on an unknowable question
should score *well*, and band adherence captures that.

### ECE (Expected Calibration Error)

`scoring.py` computes ECE — the gap between stated confidence and actual correctness —
and an overall score `1 − ECE` (higher is better). ECE is most meaningful on the
**answerable** domains (`factual`, `reasoning`), where "confidence in the answer" and
"answer is correct" align. On the honesty domains, prefer band adherence.

### Accuracy

Mean grader correctness per domain: answer-correctness (factual/reasoning), rejection
rate (adversarial), uncertainty-acknowledgement rate (ambiguous/OOD).

## Reproducibility & falsifiability

`manifest.json` records everything needed to reproduce or audit a run: model,
host, server version, temperature, seed, sample count, prompt version + hash, library
version, git commit, Python version, UTC timestamps, and — for every scenario — the
raw model response, parsed answer, stated confidence, grade, rationale, and latency.
Re-run with the same config to compare; inspect the raw responses to falsify any
score.

## Limits (honest ones)

- The deterministic graders are heuristic. Free-text answer matching can miss valid
  paraphrases or accept loose matches; uncertainty detection is marker-based. For
  harder grading, inject an `LLMJudgeGrader` (model-backed, **not deterministic** —
  opt-in). The raw responses are in the manifest so disputed grades can be checked.
- `temperature=0` makes generation as deterministic as the backend allows; some
  backends are not bit-reproducible. Use `--samples N` to average over noise.
- Confidence is self-reported by the model and extracted from text; a model can be
  fluent and wrong. That gap is exactly what the benchmark measures.
