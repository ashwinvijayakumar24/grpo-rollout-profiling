"""The E4 sync schedule, tested without vLLM: a fake sampler stands in for VLLMGeneration.

Builds a tiny-model ProfiledGRPOTrainer, then flips it onto the vLLM code path with a
fake ``vllm_generation`` that records when weights were pushed. Checks that sync
happens at steps 0, k, 2k, ... and is charged to the weight_sync bucket.
"""

import os
import sys

import pytest

from rlstudy.timing import StepTimer

pytestmark = pytest.mark.slow

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


class FakeVLLM:
    def __init__(self):
        self.synced_at = []
        self.trainer = None

    def sync_weights(self):
        self.synced_at.append(self.trainer.state.global_step)

    def generate(self, prompts, images, num_generations, profiler):
        n = len(prompts)
        return None, [[1, 2, 3]] * n, [[[-0.5], [-0.5], [-0.5]]] * n, None


@pytest.fixture
def trainer(tmp_path):
    if sys.platform == "darwin":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from rlstudy.data import as_hf_dataset
    from rlstudy.instrumented import ProfiledGRPOTrainer
    from rlstudy.reward import correctness_reward
    from rlstudy.trace import TraceWriter
    from rlstudy.train import build_grpo_config, load_config

    def make(sync_every):
        cfg = load_config(None, [f"model={TINY}", "rollout.backend=hf", "grpo.num_generations=2",
                                 "grpo.prompts_per_step=1", "grpo.max_completion_length=4"])
        t = ProfiledGRPOTrainer(
            model=TINY, reward_funcs=[correctness_reward], args=build_grpo_config(cfg, tmp_path),
            train_dataset=as_hf_dataset("train", limit=4), timer=StepTimer(),
            trace=TraceWriter(tmp_path / f"k{sync_every}", cfg), sync_every=sync_every,
        )
        fake = FakeVLLM()
        fake.trainer = t
        t.use_vllm = True
        t.vllm_generation = fake
        t._last_loaded_step = -1
        return t, fake

    return make


def drive(t, steps):
    stats = []
    for step in range(steps):
        t.state.global_step = step
        t.timer.begin_step(step)
        t._generate_single_turn([[5, 6]], None, {}, 2)
        stats.append(dict(t._step_stats))
        stats[-1]["record"] = t.timer.end_step()
        t._step_stats = {}
    return stats


@pytest.mark.parametrize("k, expected", [(1, list(range(9))), (4, [0, 4, 8]), (16, [0])])
def test_sync_happens_every_k_steps(trainer, k, expected):
    t, fake = trainer(k)
    stats = drive(t, 9)
    assert fake.synced_at == expected
    assert [s["weight_synced"] for s in stats] == [i in expected for i in range(9)]
    # Staleness: how many optimizer steps old the sampler's weights are.
    assert [s["sampler_staleness"] for s in stats] == [i - max(e for e in expected if e <= i) for i in range(9)]


def test_sync_is_charged_to_weight_sync_and_not_rollout(trainer):
    t, _ = trainer(1)
    rec = drive(t, 1)[0]["record"]
    assert rec.counts["weight_sync"] == 1
    assert rec.counts["rollout_gen"] == 1


def test_trl_guard_never_syncs_on_its_own(trainer):
    # TRL's own guard (global_step != _last_loaded_step) must be neutralized, or it
    # would sync every step regardless of k.
    t, fake = trainer(4)
    drive(t, 4)
    assert fake.synced_at == [0]
    assert t._last_loaded_step == 3
