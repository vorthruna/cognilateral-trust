# Contributing to cognilateral-trust

We welcome contributions. The library is small by design — zero dependencies, pure Python, focused on one thing: mapping confidence to trustworthy action.

## Quick Start

```bash
git clone https://github.com/vorthruna/cognilateral-trust.git
cd cognilateral-trust
uv sync
uv run pytest tests/ -q
```

## What We Need

**High-value contributions:**
- Framework integration examples (AutoGen, Semantic Kernel, DSPy, Haystack)
- Real-world usage stories (how you integrated trust scoring)
- Calibration research (does the C0-C9 tier model improve over time?)
- Language ports (TypeScript, Go, Rust)

**Always welcome:**
- Bug reports with reproduction steps
- Documentation improvements
- Test coverage for edge cases

## Guidelines

- Zero external dependencies in core (`src/cognilateral_trust/`). Framework deps only in `examples/`.
- Every change needs tests. Run `uv run pytest tests/ -q` before submitting.
- `uv run ruff check src/ tests/` must pass.
- Python 3.11+ required.
- `from __future__ import annotations` in every file.

## Design Principle

Before adding a feature, ask: **"Does this protect the person at the other end when the AI is wrong and doesn't know it?"**

If yes, ship it. If it only helps the developer but not the downstream person, it's a nice-to-have.
