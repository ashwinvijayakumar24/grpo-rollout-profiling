# grpo-rollout-profiling

Where does wall-clock time go in GRPO post-training of a small LLM, and what are the
bottlenecks in the rollout (answer-generation) path?

This is a systems study, not new RL research. It runs a small GRPO loop on GSM8K
math with an exact-match reward, times every stage of every training step, and
measures how the time split changes with group size, generation length, and
weight-sync frequency.

Status: planning. See [PLAN.md](PLAN.md).
