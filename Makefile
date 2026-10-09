# One-command entry points. GPU targets run on a CUDA machine (PACE: submit with
# `make submit-<exp>` from the login node). See PLAN.md for what each experiment asks.
PY ?= python
SPECS = configs/experiments

.PHONY: test test-all smoke-local baseline analyze figures

test:            ## fast unit tests (no model download)
	$(PY) -m pytest -q -m "not slow"

test-all:        ## includes the tiny-model end-to-end runs
	$(PY) -m pytest -q

smoke-local:     ## 3 steps, tiny random model, HF generate: checks plumbing on any machine
	$(PY) -m rlstudy.train --out scratch/smoke_local \
	  --set model=trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --set rollout.backend=hf \
	  --set grpo.max_steps=3 --set grpo.prompts_per_step=2 --set grpo.num_generations=4 \
	  --set grpo.max_completion_length=16 --set data.limit=16

baseline:        ## the instrumented baseline loop (E1 timing arm, 3 reps) on a CUDA GPU
	$(PY) -m rlstudy.experiment $(SPECS)/E1_breakdown.yaml --arms baseline

analyze:         ## per-run summaries, per-arm stats, summary.md tables
	$(PY) -m rlstudy.analyze results/*/

figures:         ## every writeup figure that has results
	$(PY) -m rlstudy.figures

submit-%:        ## on the PACE login node: sbatch one experiment spec
	sbatch scripts/slurm/run_experiment.sbatch $(SPECS)/$*$(if $(filter smoke,$*),,_*).yaml
