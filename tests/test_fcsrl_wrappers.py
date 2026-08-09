# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test the FCSRL-style training-setup wrappers."""

import numpy as np
import pytest  # pylint: disable=import-error

import helpers
import safety_gymnasium
from safety_gymnasium.utils.normalizer import MeanStdNormalizer
from safety_gymnasium.wrappers import (
    SafeActionRepeat,
    SafeGoalMetTerminal,
    SafeNormalizeCost,
    SafeNormalizeObservation,
    SafeNormalizeReward,
    SafePixelObservation,
)


ENV_ID = 'SafetyPointGoal1-v0'


# ==============================================================================
# SafeActionRepeat
# ==============================================================================


@helpers.parametrize(n_repeat=[1, 2, 4])
def test_action_repeat_episode_length(n_repeat):
    """An episode covers the same simulator steps regardless of the repeat."""
    env = SafeActionRepeat(safety_gymnasium.make(ENV_ID), n_repeat)
    env.reset(seed=0)

    agent_steps, sim_steps = 0, 0
    while True:
        _, _, _, terminated, truncated, info = env.step(env.action_space.sample())
        agent_steps += 1
        sim_steps += info['sim_steps']
        if terminated or truncated:
            break

    assert sim_steps == 1000, f'Expected 1000 simulator steps, got {sim_steps}.'
    assert agent_steps == 1000 // n_repeat


def test_action_repeat_aggregates_reward_and_cost():
    """Reward and cost are the sums over the repeated simulator steps."""
    action = np.zeros(safety_gymnasium.make(ENV_ID).action_space.shape, dtype=np.float64)

    unwrapped = safety_gymnasium.make(ENV_ID)
    unwrapped.reset(seed=0)
    expected_reward, expected_cost, expected_hazards = 0.0, 0.0, 0.0
    for _ in range(4):
        _, reward, cost, _, _, info = unwrapped.step(action)
        expected_reward += reward
        expected_cost += cost
        expected_hazards += info['cost_hazards']

    repeated = SafeActionRepeat(safety_gymnasium.make(ENV_ID), 4)
    repeated.reset(seed=0)
    _, reward, cost, _, _, info = repeated.step(action)

    assert reward == pytest.approx(expected_reward)
    assert cost == pytest.approx(expected_cost)
    assert info['cost_hazards'] == pytest.approx(expected_hazards)
    assert info['sim_steps'] == 4


def test_action_repeat_breaks_early_on_truncation():
    """A repeat that straddles the time limit stops at the boundary."""
    env = SafeActionRepeat(safety_gymnasium.make(ENV_ID, max_episode_steps=10), 4)
    env.reset(seed=0)

    step_counts = []
    while True:
        _, _, _, terminated, truncated, info = env.step(env.action_space.sample())
        step_counts.append(info['sim_steps'])
        if terminated or truncated:
            break

    assert step_counts == [4, 4, 2], f'Expected [4, 4, 2] simulator steps, got {step_counts}.'
    assert truncated


def test_action_repeat_rejects_invalid_n_repeat():
    """A non-positive repeat is a configuration error."""
    with pytest.raises(ValueError, match='positive integer'):
        SafeActionRepeat(safety_gymnasium.make(ENV_ID), 0)


# ==============================================================================
# SafeGoalMetTerminal
# ==============================================================================


def _run_until_goal(env, max_steps=1000):
    """Step until ``goal_met`` appears, returning that step's outputs."""
    for _ in range(max_steps):
        outputs = env.step(env.action_space.sample())
        if outputs[5].get('goal_met', False):
            return outputs
    raise AssertionError('The goal was never reached.')


def test_goal_met_is_pseudo_terminal_not_terminal():
    """Reaching the goal flags the pseudo-terminal but does not end the episode."""
    env = SafeGoalMetTerminal(safety_gymnasium.make(ENV_ID))
    env.reset(seed=0)
    # Enlarge the goal so that a random policy reaches it quickly.
    env.unwrapped.task.goal.size = 2.5

    _, _, _, terminated, truncated, info = _run_until_goal(env)

    assert info['pseudo_terminated'] is True
    assert terminated is False, 'Goal achievement must not terminate the episode.'
    assert truncated is False

    # The episode really does continue.
    env.step(env.action_space.sample())


def test_pseudo_terminal_survives_action_repeat():
    """The flag is OR-reduced across the repeat, in either wrapper order."""
    inside = SafeActionRepeat(SafeGoalMetTerminal(safety_gymnasium.make(ENV_ID)), 4)
    outside = SafeGoalMetTerminal(SafeActionRepeat(safety_gymnasium.make(ENV_ID), 4))

    for env in (inside, outside):
        env.reset(seed=0)
        env.unwrapped.task.goal.size = 2.5
        _, _, _, terminated, _, info = _run_until_goal(env, max_steps=250)
        assert info['pseudo_terminated'] is True
        assert terminated is False


def test_pseudo_terminal_defaults_to_false():
    """An ordinary step is not a pseudo-terminal."""
    env = SafeGoalMetTerminal(safety_gymnasium.make(ENV_ID))
    env.reset(seed=0)
    _, _, _, _, _, info = env.step(env.action_space.sample())
    assert info['pseudo_terminated'] is False


# ==============================================================================
# Normalization
# ==============================================================================


def test_normalizer_standardizes():
    """A normalized sample has roughly zero mean and unit variance."""
    rng = np.random.default_rng(0)
    normalizer = MeanStdNormalizer(read_only=False)

    samples = rng.normal(loc=5.0, scale=2.0, size=(4096, 3))
    for start in range(0, len(samples), 64):
        normalized = normalizer(samples[start : start + 64])

    assert normalized.shape == (64, 3)
    assert normalizer.rms.mean == pytest.approx(5.0, abs=0.1)
    assert np.sqrt(normalizer.rms.var) == pytest.approx(2.0, abs=0.1)


def test_normalizer_clips():
    """Extreme values are clipped to the configured range."""
    normalizer = MeanStdNormalizer(clip=2.0, read_only=False)
    normalizer(np.random.default_rng(0).normal(size=(512, 1)))
    normalizer.set_read_only()

    assert normalizer(np.array([[1e6]]))[0, 0] == pytest.approx(2.0)
    assert normalizer(np.array([[-1e6]]))[0, 0] == pytest.approx(-2.0)


def test_normalizer_read_only_freezes_statistics():
    """A read-only normalizer normalizes without moving its statistics."""
    normalizer = MeanStdNormalizer(read_only=False)
    normalizer(np.zeros((8, 2)))
    normalizer.set_read_only()

    before = normalizer.state_dict()
    normalizer(np.full((8, 2), 100.0))
    after = normalizer.state_dict()

    assert after['mean'] == pytest.approx(before['mean'])
    assert after['count'] == before['count']


def test_normalizer_state_dict_round_trip():
    """Saved statistics restore into a fresh normalizer."""
    source = MeanStdNormalizer(read_only=False)
    source(np.random.default_rng(0).normal(size=(256, 4)))
    source.set_read_only()

    restored = MeanStdNormalizer()
    restored.load_state_dict(source.state_dict())

    sample = np.random.default_rng(1).normal(size=(8, 4))
    assert restored(sample) == pytest.approx(source(sample))


def test_normalizer_unnormalize_inverts():
    """``unnormalize`` inverts ``__call__`` when no clipping occurred."""
    normalizer = MeanStdNormalizer(read_only=False)
    samples = np.random.default_rng(0).normal(loc=3.0, size=(256, 2))
    normalizer(samples)
    normalizer.set_read_only()

    sample = samples[:4]
    assert normalizer.unnormalize(normalizer(sample)) == pytest.approx(sample)


@helpers.parametrize(
    wrapper=[SafeNormalizeObservation, SafeNormalizeReward, SafeNormalizeCost],
)
def test_normalize_wrappers_step(wrapper):
    """All three normalization wrappers reset and step without error."""
    env = wrapper(safety_gymnasium.make(ENV_ID))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(4):
        obs, reward, cost, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert np.isfinite(cost)


def test_normalize_observation_respects_clip_and_freeze():
    """Observations stay within the clip range, and freezing stops the updates."""
    env = SafeNormalizeObservation(safety_gymnasium.make(ENV_ID), clip=5.0)
    obs, _ = env.reset(seed=0)
    assert np.all(np.abs(obs) <= 5.0)
    assert env.observation_space.low.min() == pytest.approx(-5.0)
    assert env.observation_space.high.max() == pytest.approx(5.0)

    for _ in range(8):
        env.step(env.action_space.sample())

    env.freeze()
    before = env.get_stats()
    for _ in range(8):
        env.step(env.action_space.sample())
    after = env.get_stats()

    assert after['count'] == before['count']
    assert after['mean'] == pytest.approx(before['mean'])

    env.unfreeze()
    env.step(env.action_space.sample())
    assert env.get_stats()['count'] > before['count']


def test_normalize_observation_shares_statistics():
    """A training and an evaluation env can share one normalizer."""
    train_env = SafeNormalizeObservation(safety_gymnasium.make(ENV_ID))
    eval_env = SafeNormalizeObservation(safety_gymnasium.make(ENV_ID))
    eval_env.normalizer = train_env.normalizer
    eval_env.freeze()

    train_env.reset(seed=0)
    for _ in range(8):
        train_env.step(train_env.action_space.sample())

    eval_env.reset(seed=0)
    assert (
        eval_env.get_stats()['count'] == train_env.get_stats()['count']
    ), 'Frozen evaluation env must not update the shared statistics.'


def test_normalize_observation_rejects_dict_space():
    """Pixel and Dict observation spaces are not supported."""
    with pytest.raises(TypeError, match='Box observation space'):
        SafeNormalizeObservation(safety_gymnasium.make('SafetyPointGoal1Vision-v0'))


# ==============================================================================
# SafePixelObservation
# ==============================================================================


def test_pixel_observation_shape_and_range():
    """Pixels are HWC float32 in [0, 1]."""
    env = SafePixelObservation(safety_gymnasium.make(ENV_ID), size=(64, 64))

    obs, _ = env.reset(seed=0)
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    assert env.observation_space.contains(obs)

    obs, _, _, _, _, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)


@helpers.parametrize(size=[(32, 32), (48, 64)])
def test_pixel_observation_honours_size(size):
    """The frame size is taken as (height, width)."""
    env = SafePixelObservation(safety_gymnasium.make(ENV_ID), size=size)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (*size, 3)


def test_pixel_observation_renders_once_per_agent_step():
    """Placed outside the repeat, the camera renders once per agent step, not four times."""
    env = SafePixelObservation(
        SafeActionRepeat(SafeGoalMetTerminal(safety_gymnasium.make(ENV_ID)), 4),
        size=(64, 64),
    )
    env.reset(seed=0)

    task = env.unwrapped.task
    original_render = task.render
    calls = []

    def counting_render(*args, **kwargs):
        calls.append(1)
        return original_render(*args, **kwargs)

    task.render = counting_render
    try:
        for _ in range(5):
            env.step(env.action_space.sample())
    finally:
        task.render = original_render

    assert len(calls) == 5, f'Expected 5 renders for 5 agent steps, got {len(calls)}.'


def test_pixel_observation_requires_a_task():
    """Velocity environments have no task to render from."""
    with pytest.raises(TypeError, match='requires an environment with a `task`'):
        SafePixelObservation(safety_gymnasium.make('SafetyHalfCheetahVelocity-v1'))


# ==============================================================================
# Determinism of the full stack
# ==============================================================================


def test_full_stack_is_deterministic():
    """The same seed and actions give the same trajectory."""

    def rollout():
        env = SafeActionRepeat(SafeGoalMetTerminal(safety_gymnasium.make(ENV_ID)), 4)
        obs, _ = env.reset(seed=7)
        rng = np.random.default_rng(7)
        trajectory = [obs]
        for _ in range(16):
            obs, reward, cost, _, _, _ = env.step(
                rng.uniform(
                    env.action_space.low,
                    env.action_space.high,
                ),
            )
            trajectory.append((obs, reward, cost))
        return trajectory

    first, second = rollout(), rollout()
    assert first[0] == pytest.approx(second[0])
    for (obs_a, rew_a, cost_a), (obs_b, rew_b, cost_b) in zip(
        first[1:],
        second[1:],
        strict=True,
    ):
        assert obs_a == pytest.approx(obs_b)
        assert rew_a == pytest.approx(rew_b)
        assert cost_a == pytest.approx(cost_b)
