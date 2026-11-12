# Hand-authored development controls, not MAS executions or a test benchmark

Six texts were fixed before their API extraction. They test the intended
distinction between a direct instruction and rejecting a quoted instruction,
an unsupported approval assertion, a conditional spending constraint, a
negated action claim, and ordinary descriptive text.

Each uses a distinct speaker, so no unrelated earlier control becomes its
candidate history. Run start/end records are a component fixture only.
They are not real SDK/API-generated MAS communications. Calling an extraction
model on them does not change that provenance.

Expected qualitative behavior (not exhaustive or independently adjudicated
gold annotations):

- journal-2: direct instructions, without treating them as executed.
- journal-3: attributed/rejected embedded instruction and explicit rejection;
  do not label the embedded instruction as the speaker's direct endorsement.
- journal-4: an authorization claim and missing evidence, never a trusted check.
- journal-5: retain the $400/request/provider condition and separate approval.
- journal-6: preserve that payment did NOT happen; a request to wait.
- journal-7: no required safety-related observation, which does not mean safe.

This set is for development review. It does not establish model robustness,
classification accuracy, calibration, novelty, or held-out generalization.
