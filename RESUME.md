# Resume bullets (draft)

> **Status: placeholders.** Each bullet gets its numbers only after the run that
> produces them exists in `results/`. Every number will carry its source path below.

- Built an instrumented GRPO post-training loop (TRL + vLLM, Qwen2.5-1.5B, GSM8K
  exact-match rewards) that attributes every step's wall clock to rollout, reward,
  loss/backward, optimizer, and weight sync with GPU-synchronized timers; found
  rollout generation takes **TODO%** of step time (E1) …
- Profiled the trainer→sampler weight-sync path: **TODO s** per sync, GPU busy only
  **TODO%** of the sync window (E4); … TODO.

## Sources

| Number | File |
|---|---|
| TODO | TODO |
