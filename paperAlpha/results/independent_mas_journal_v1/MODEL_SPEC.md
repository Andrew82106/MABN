# Two-line MAS risk monitor — inspectable specification

This is the active development candidate evaluated in `metrics.json`. In the paper,
call it a **knowledge-compiled conditional BN factorization**: the graph, named
conditional factors and noisy-OR composition are explicit and inspectable, but this
is not causal discovery and not a claim that the protected LLM internally uses the
same graph.

## Inputs

- Normative line: public `workflow_dag.nodes` and `workflow_dag.edges`.
- Runtime line: public `events` before the declared scoring horizon. The event parser uses actor/target, event type, data class, edge kind, edge confidence, tool class and permission mismatch.
- Never used: `label`, `external_effect`, `hidden_fields`, `label_source`, `mechanism`, `scenario_family`.

## Layer 1: workflow BN

For each episode, the workflow evidence vector is

`[agent_count, workflow_edge_count, source_count, privileged_sink_count, normative_source_sink_path, normative_short_path, max_out_degree, max_in_degree, mean_out_degree, expected_edge_missingness]`.

For each binary evidence factor `f`, training estimates

`q_f = (N(risk and f) + 1) / (N(f) + 2)`.

The workflow score is a noisy-OR over present factors:

`p_W = 1 - product_f(1 - q_f)`.

This is an evidence score, not a causal probability and not a claim that the workflow DAG is discovered from the model internals.

## Layer 2: runtime BN

The runtime parser creates explicit message/delegation edges. Each edge has an inspectable feature vector for carrier, untrusted provenance, unexpectedness, confidence, unknownness, delegation and workflow membership. A learned sigmoid gate scores each edge, and a noisy-OR propagates state from untrusted-input sources through observed edges. The privileged-sink risk is a sigmoid head over propagated path evidence, local evidence, Layer-1 node evidence, asserted path evidence and path distance. Unknown/redacted edges are evaluated under point, lower and upper observation policies; they are not silently treated as safe.

## Fusion

## Explicit factor graph

The released monitor has the following inspectable dependency DAG (arrows are
model dependencies, not causal claims): `W_f -> W_R`,
`R_edge -> R_path -> R_R`, `W_f -> R_R`, and
`(W_R, R_R, runtime summaries) -> F`.

For each workflow factor, `P(W_f=1)=q_f`; the workflow node uses
`P(W_R=1 | W_1,...,W_m)=1-prod_f(1-W_f q_f)`. Each runtime edge has the
explicit sigmoid CPD `P(R_edge=1 | x_edge)=sigmoid(theta^T x_edge)`, path
propagation is the declared noisy-OR over observed edges, and the sink node is
the documented sigmoid head over path/local/workflow evidence. The fusion head
is fit only on inner-OOF line scores and declared summaries. Every active
factor, edge, parameter and missingness policy can therefore be inspected or
replayed; the graph is knowledge-compiled rather than discovered from hidden
LLM state.

The outer test fold is never used to fit a factor or threshold. Within each outer training fold, three grouped inner folds create OOF `p_W` and `p_R` scores. A logistic fusion head then receives the raw workflow/runtime summaries plus these two OOF BN scores. The outer test episode receives scores from factors fit on the full outer training fold. The classification threshold is also selected from inner-OOF fusion scores.

The final monitor exposes both line scores, fusion score, active factors, observed edges, unknown evidence and the interval/abstention policy. It is therefore white-box at the monitor level, while the protected API LLM remains black-box.

## Claim boundary

The monitor performs system-level risk assessment from observable traces. It does not perform causal discovery, inspect the protected model's weights, or guarantee absolute safety.
