# Preparation repair history

## Attempt 1 — preserved, rejected before GPU

The first CPU preparation retained 786 of 4,381 conflict spans by replacing the
entire metadata `Generated:` phrase with the entire source-aligned `Original:`
phrase. A manual quality audit then found that many Data2txt corrections were
structured fields such as `"OutdoorSeating": false`. Direct insertion produced
malformed claims such as `and OutdoorSeating': False`. Some natural-language
fields also changed syntax rather than a factual slot.

No GPU command used this data. The attempt remains unchanged for audit:

| File | SHA256 |
|---|---|
| `PROTOCOL.md` | `64f91ca90d1f53e71dfedf2e402c6b3bc3793bf43bb086e9580320f5d9383f0a` |
| `triples.jsonl` | `f73db9bfb3625efcd18973cf370289b07a371b9ec4c7264cfa10cc7cd1358e5f` |
| `span_audit.jsonl` | `92f7bb1d5216a4f0544dc9ca812276bee790a83e652b3fcb2683df8d57b1720a` |
| `arrays.npz` | `925956aa24a004eb3590aca77a729715f69977e722be75b63a8a9143af196f3b` |
| `preparation_complete.json` | `6861771ab3b70c08d7ed57e55f8ba98c5c770156f13c88c9359c1dd4ad8232ba` |
| `manifest.json` | `20bd89038cd63aeb3314be1062defa55ade156702b71060ac90ef50e58b4bcc1` |

## Repair v2 — stopped at data audit

Repair v2 was reconsidered before any dataset, runner, or GPU artifact was
created. An independent full-corpus and fixed 100-item review showed that an
`Original:` field is usually an annotator explanation or evidence hint, not a
grammatically interchangeable correction. Only 14/100 reviewed items were safe
direct pairs. The frozen automatic prefilter found 272 span candidates (263
unique pairs), or 150 candidates (146 unique pairs) when the answer had only one
released error; these remain review candidates, not gold labels.

Consequently, `repair_v2` is audit-only and is not a training candidate. No
threshold was relaxed, attempt 1 is not a fallback training set, and no GPU was
run. See `STOP_REPORT.md` and the independent feasibility audit referenced there.
