# Same-episode API-to-monitor cost audit

- Shared episodes: **10 / 80 API episodes
- Serial service-time estimate p50/p95: **47.386 / 83.429 s**
- Monitor local semantic+graph+BN p50/p95: **0.691 / 0.996 ms**
- Token totals: API in/out 8223/7576; semantic in/out 8192/14821

The report joins only episode IDs present in both frozen ledgers. It is a cost measurement, not an accuracy result; failed requests and semantic attempts remain visible.
