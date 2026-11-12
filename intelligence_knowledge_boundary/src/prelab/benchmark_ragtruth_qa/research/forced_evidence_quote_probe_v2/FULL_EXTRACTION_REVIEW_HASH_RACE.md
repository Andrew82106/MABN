# V2 full-extraction review-hash race

The first full-extraction invocation passed the runtime gate while the GPU
smoke reviewer was still replacing a draft PASS receipt with its final PASS
receipt.  The review JSON changed from draft SHA-256
`d8e6486adf047e83cd85f760fd398e84f0a2e77252e42bd180e701df9505f456`
to final SHA-256
`e84324a9c77622e423995555ea754ddd094a325aadce247a0f45d83d1ef65649`.
Both receipts represented the same PASS decision, but the invocation was
interrupted during claim 17 so that the stable final receipt could be checked
again before continuing.

Sixteen complete atomic records had already been committed.  They are retained:
their signatures bind the unchanged runner, runtime, input row, model, protocol,
plan, numerical protocol, and feature computation.  The partial seventeenth
claim was never committed.  The next invocation must re-run every gate against
the final review receipt, validate the 16 committed records, and resume only the
remaining records.  No record, sample, feature, threshold, protocol, runner, or
label was changed, and no gold/calibration/test data was opened.
