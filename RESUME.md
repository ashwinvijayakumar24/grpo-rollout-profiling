# Resume bullets (draft)

- Built an instrumented GRPO post-training loop (TRL + vLLM sharing one H100,
  Qwen2.5-1.5B, GSM8K exact-match rewards) that attributes every step's wall clock to
  rollout, scoring, loss/backward, optimizer, and weight sync at 0.4% overhead; found
  trainer-side compute, not generation, dominated (57% vs 33% of a 2.27 s step) and
  that rollout time tracks the longest sample (40% decode-slot occupancy at a
  512-token cap).
- Profiled the trainer→sampler weight-sync path with kernel-level traces: each sync
  took 0.18 s while the GPU was busy only 2% of it across 338 per-tensor copies;
  quantified the sync-frequency trade-off (syncing every 16 steps cut step time 7.6%)
  and showed driver utilization counters misreport sub-second stages (80% vs 2%).

## Sources

| Number | File |
|---|---|
| 0.4% overhead | `results/E0_overhead/overhead.json` (`mean_diff_pct` 0.401) |
| 57% / 33% / 2.27 s | `results/E1_breakdown/baseline/arm_summary.json` (`buckets.*.share`, `step_time_mean_s` 2.266) |
| 40% slot occupancy at 512 | `results/E3_gen_length/len512/arm_summary.json` (`slot_occupancy` 0.403) |
| 0.18 s per sync | `results/E1_breakdown/baseline/arm_summary.json` (`sync.mean_s_per_sync` 0.178) |
| GPU busy 2% during sync | `results/E1_breakdown/baseline_profiled/rep0/profile_busy.json` (`weight_sync.busy_frac` 0.020) |
| 338 tensors | `results/env/model_param_count.json` |
| 7.6% from syncing every 16 steps | `results/E4_sync_freq/{sync1,sync16}/arm_summary.json` (2.287 → 2.113 s) |
| NVML 80% vs profiler 2% | `results/E1_breakdown/baseline/arm_summary.json` (`nvml_util_by_span.weight_sync` 79.9) and the profiler file above |
