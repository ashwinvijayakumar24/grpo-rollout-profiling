# Future work (out of scope for this study)

Ideas that came up and were deliberately not pursued, so the study stays small.

- **SGLang as a rollout backend.** TRL has no SGLang path, and SGLang 0.5.20 needs
  CUDA 13 and its own environment. Dropped from E5. (DECISIONS.md)
- **Adapter-only weight sync for LoRA.** TRL merges the adapter and sends all base
  weights to vLLM on every sync. Sending only the adapter (as veRL can via
  `add_lora`) should be much cheaper. Worth building only after E4 measures how big
  the sync cost actually is. (NOTES.md section 2)
- **Using the from-scratch `llm_serving_layer` as the rollout engine.** A possible E5
  arm. Only if E1–E4 finish early.
