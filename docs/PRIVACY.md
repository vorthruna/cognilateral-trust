# Privacy

*Last updated: 2026-04-28*

`cognilateral-trust` is a Python library that runs on your machine. Installing and using it does not transmit any data to any service operated by the maintainer.

## What I collect

Nothing. The library has zero runtime dependencies and makes no network calls. PyPI logs downloads when you `pip install`; that is between you and PyPI.

## Persistence

If you opt into the persistence helpers (`JSONLPredictionStore`, `JSONLAccountabilityStore`), records are written to local files at the paths you specify. The library does not transmit them. Whether those files are then synced, backed up, or replicated is up to whatever tools you have configured at the OS or cloud-storage level — the library has no view into that.

## Contact

`hello@cognilateral.com` for any questions about how the library handles data.
