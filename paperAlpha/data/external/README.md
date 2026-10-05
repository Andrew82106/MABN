# External data provenance

Raw external releases are intentionally kept out of Git. The local audit used:

- MAST/MAD full release: `data/external/MAST/MAD_full_dataset.json`
- ATBench test release: `data/external/ATBench/test.json`
- AgentLeak release: `data/external/AgentLeak/`
- A2ASecBench release: `data/external/A2ASecBench/`
- Who-and-When release: `data/external/Who_and_When/`

The exact source URLs, revisions and file hashes used by the audits are stored
in each generated report under `results/submission/development/` and in the
submission protocol. Re-run the corresponding adapter only after downloading
the named release; do not commit API keys or raw provider credentials.
