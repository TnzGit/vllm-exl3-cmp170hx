# C2/C4 frozen-input manifest provenance

- Hardware runner source: `fd0d6dca871ca5a512393b48e334fe26a5086e73`.
- Manifest: `c2c4-manifest-fd0d6dc.json`, SHA-256
  `0522e679e0d7daac5a735500e1109e331a6c877f32ea5fd8ae93ee42ea6c96ca`.
- Model-tree SHA-256: `5914768b619464b6d9d7262c0f5395a2ea3e93e19677f2c7ac97d830afab5936`.
- Tokenizer-tree SHA-256: `cf5af24b6d0fa88d0667c4561f8c185be99c3509bfe82f4972d5fe940a3095e5`.
- The manifest's `model_revision_sha256` field contains the local
  `model.safetensors.index.json` SHA-256
  (`37d098b99467c756094e2a9f089a7a92b119506daee77ad692767f8236b570ca`),
  **not** a proven upstream Hub commit. The full model-tree hash is the
  authoritative byte identity for this experiment.
- Archived parents are the eight committed `ctx16000/` and `ctx80000/` turn
  files under `c2c4-source-prompts/`. The generator records parent file and
  token-ID hashes per request. These benchmark inputs are explicitly derived
  from those turns, not historical byte-identical replays.
- `c4_32k` is blocked because its historical input IDs are unavailable; the
  manifest contains no synthetic substitute.

Generation used the pinned model's local tokenizer, the fixed 2048 effective
auto batching budget, and the committed generator at the runner source SHA.
The standalone matched-load dry-run and one-cell wrapper dry-run passed before
the first hardware cell. Hardware results are recorded separately; this file
does not claim any GPU result.
