# Unified topology queue (local fixture)

`paperalpha_runtime.unified_topology_queue` is a small transport contract for
the four existing runner shapes:

| topology | shape | existing analogue |
|---|---|---|
| `chain` | one predecessor at each stage | `fixed_workflow` / review stage |
| `fork` | one root, independent leaves | `branched_authority` |
| `join` | independent leaves, one sink | `mas_purchase` batch |
| `review` | work stage followed by reviewer | `harnessaudit` review boundary |

The queue executes each submitted node once. Handler exceptions become a
terminal `status: "failed"` record; no retry or filtering occurs. A handler
receives only the completed parent records required by the declared edges.

`QueueRun.monitor_projection()` is the public view. It includes topology,
request/response observations, statuses and failure evidence, while recursively
removing evaluator-only keys (`label`, `expected`, `ground_truth`, `oracle`,
etc.). `evaluator=` receives a private copy of all records and its result is
available only from `evaluation_view()`.

The CLI writes `monitor.json` and (unless `--no-evaluator`) `evaluator.json`
under the ignored `results/` tree. It uses a deterministic local fixture
handler; it is not a real external API collection job.

