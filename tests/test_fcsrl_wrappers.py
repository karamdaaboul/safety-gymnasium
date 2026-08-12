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

import gymnasium
import numpy as np
import pytest  # pylint: disable=import-error

import helpers
import safety_gymnasium
from safety_gymnasium.utils.normalizer import MeanStdNormalizer
from safety_gymnasium.wrappers import (
    SafeActionRepeat,
    SafeCostLimitCurriculum,
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
    """A training and an evaluation env can share one normalizer.

    Freezing the evaluation env must not stop the training env from updating the
    shared statistics, and the evaluation env must not contribute to them.
    """
    train_env = SafeNormalizeObservation(safety_gymnasium.make(ENV_ID))
    eval_env = SafeNormalizeObservation(safety_gymnasium.make(ENV_ID))
    eval_env.normalizer = train_env.normalizer
    eval_env.freeze()

    train_env.reset(seed=0)
    start = train_env.get_stats()['count']
    for _ in range(8):
        train_env.step(train_env.action_space.sample())
    after_training = train_env.get_stats()['count']

    assert after_training == pytest.approx(
        start + 8,
    ), 'Training env stopped updating the statistics when the eval env was frozen.'
    assert not np.allclose(
        train_env.get_stats()['mean'],
        0.0,
    ), 'Statistics never moved, so normalization silently degenerated to clipping.'

    eval_env.reset(seed=0)
    for _ in range(8):
        eval_env.step(eval_env.action_space.sample())

    assert train_env.get_stats()['count'] == pytest.approx(
        after_training,
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
# SafeCostLimitCurriculum
# ==============================================================================


def _curriculum_env(max_episode_steps=10, **kwargs):
    """Build a short-episode curriculum env; kwargs override the defaults."""
    params = {
        'initial_cost_limit': 100.0,
        'min_cost_limit': 10.0,
        'decrement': 20.0,
        'success_threshold': 0.0,
        'window': 1,
    }
    params.update(kwargs)
    return SafeCostLimitCurriculum(
        safety_gymnasium.make(ENV_ID, max_episode_steps=max_episode_steps),
        **params,
    )


def _run_episodes(env, n_episodes):
    """Run random episodes, resetting between them; return each boundary step's info."""
    boundary_infos = []
    for _ in range(n_episodes):
        while True:
            _, _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                boundary_infos.append(info)
                env.reset()
                break
    return boundary_infos


def test_cost_limit_curriculum_rejects_invalid_args():
    """A misordered or degenerate curriculum is a configuration error."""
    with pytest.raises(ValueError, match='min_cost_limit must not exceed'):
        _curriculum_env(initial_cost_limit=10.0, min_cost_limit=20.0)
    with pytest.raises(ValueError, match='decrement must be positive'):
        _curriculum_env(decrement=0.0)
    with pytest.raises(ValueError, match='success_threshold must be non-negative'):
        _curriculum_env(success_threshold=-1.0)
    with pytest.raises(ValueError, match='window must be a positive integer'):
        _curriculum_env(window=0)


def test_cost_limit_reported_on_reset_and_step():
    """The current limit appears in info on reset and step, and nothing else changes."""
    action = np.zeros(safety_gymnasium.make(ENV_ID).action_space.shape, dtype=np.float64)

    unwrapped = safety_gymnasium.make(ENV_ID)
    unwrapped.reset(seed=0)
    _, expected_reward, expected_cost, _, _, _ = unwrapped.step(action)

    env = SafeCostLimitCurriculum(
        safety_gymnasium.make(ENV_ID),
        initial_cost_limit=100.0,
        min_cost_limit=10.0,
        decrement=20.0,
        success_threshold=1.5,
        window=20,
    )
    _, info = env.reset(seed=0)
    assert info['cost_limit'] == pytest.approx(100.0)
    assert env.cost_limit == pytest.approx(100.0)

    _, reward, cost, terminated, truncated, info = env.step(action)
    assert info['cost_limit'] == pytest.approx(100.0)
    assert reward == pytest.approx(expected_reward), 'The wrapper must not alter the reward.'
    assert cost == pytest.approx(expected_cost), 'The wrapper must not alter the cost.'
    assert terminated is False
    assert truncated is False


def test_cost_limit_decays_on_success():
    """Each successful window steps the limit down by the decrement."""
    env = _curriculum_env()  # window=1, threshold=0: every episode advances the stage
    env.reset(seed=0)

    boundary_infos = _run_episodes(env, 3)

    assert env.cost_limit == pytest.approx(40.0), 'Expected 100 - 3 * 20 after 3 episodes.'
    assert boundary_infos[0]['cost_limit'] == pytest.approx(
        80.0,
    ), 'The boundary step must already report the newly decayed limit.'


def test_cost_limit_holds_below_threshold():
    """An unreachable threshold leaves the limit at its initial value."""
    env = _curriculum_env(success_threshold=100.0)
    env.reset(seed=0)

    _run_episodes(env, 3)

    assert env.cost_limit == pytest.approx(100.0)
    assert env.window_mean is not None, 'Completed episodes must populate the window.'
    assert env.window_mean < 100.0


def test_cost_limit_floors_at_min():
    """An overshooting decrement clamps to the minimum, where the curriculum goes inert."""
    env = _curriculum_env(initial_cost_limit=25.0, min_cost_limit=5.0, decrement=10.0)
    env.reset(seed=0)

    limits = [info['cost_limit'] for info in _run_episodes(env, 4)]

    assert limits == pytest.approx([15.0, 5.0, 5.0, 5.0])


def test_window_clears_after_decrement():
    """Each stage is judged on fresh episodes, not a still-sliding window."""
    env = _curriculum_env(window=2)
    env.reset(seed=0)

    _run_episodes(env, 5)

    # Decrements after episodes 2 and 4 only; a window that kept sliding would
    # decrement after every episode from the second onwards (4 decrements).
    assert env.cost_limit == pytest.approx(60.0), 'Expected 100 - 2 * 20 after 5 episodes.'


def test_episode_goals_counts_and_reset_discards():
    """Goals are counted within an episode and a mid-episode reset discards them."""
    env = SafeCostLimitCurriculum(
        safety_gymnasium.make(ENV_ID),
        initial_cost_limit=100.0,
        min_cost_limit=10.0,
        decrement=20.0,
        success_threshold=100.0,
        window=1,
    )
    env.reset(seed=0)
    # Enlarge the goal so that a random policy reaches it quickly.
    env.unwrapped.task.goal.size = 2.5

    _run_until_goal(env)
    assert env.episode_goals == 1
    assert env.window_mean is None, 'No episode has completed yet.'

    _, info = env.reset()
    assert env.episode_goals == 0
    assert env.window_mean is None, 'A partial episode must not be recorded.'
    assert env.cost_limit == pytest.approx(100.0)
    assert info['cost_limit'] == pytest.approx(100.0)


def test_cost_limit_state_round_trip():
    """Saved curriculum state restores into a fresh wrapper."""
    source = _curriculum_env(window=2)
    source.reset(seed=0)
    _run_episodes(source, 3)
    for _ in range(4):  # a few steps into the next episode
        source.step(source.action_space.sample())

    restored = _curriculum_env(window=2)
    restored.set_state(source.get_state())

    assert restored.cost_limit == pytest.approx(source.cost_limit)
    assert restored.episode_goals == source.episode_goals
    assert restored.get_state() == source.get_state()


class _FixedCostEnv(gymnasium.Env):
    """Minimal 6-tuple env with a settable per-step cost, for exact gate tests.

    Real rollouts give a cost that is random and usually zero over a short episode,
    which cannot pin down a threshold comparison.
    """

    def __init__(self, cost_per_step, episode_len=5):
        self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), dtype=np.float64)
        self.cost_per_step = cost_per_step
        self._episode_len = episode_len
        self._step = 0

    def reset(self, **kwargs):  # pylint: disable=unused-argument
        self._step = 0
        return np.zeros(1), {}

    def step(self, action):  # pylint: disable=unused-argument
        self._step += 1
        # goal_met on every step, so the goal gate is never what blocks these tests.
        return np.zeros(1), 0.0, self.cost_per_step, False, self._step >= self._episode_len, {'goal_met': True}


def _cost_gated_env(cost_per_step, **kwargs):
    """Curriculum over `_FixedCostEnv`; episode cost is 5 x cost_per_step."""
    params = {
        'initial_cost_limit': 100.0,
        'min_cost_limit': 10.0,
        'decrement': 20.0,
        'success_threshold': 0.0,
        'window': 1,
        'gate_on_cost': True,
    }
    params.update(kwargs)
    return SafeCostLimitCurriculum(_FixedCostEnv(cost_per_step), **params)


def test_cost_gate_advances_when_within_budget():
    """Episode cost under the limit leaves the goal gate in charge."""
    env = _cost_gated_env(cost_per_step=1.0)  # episode cost 5, well under 100
    env.reset()

    _run_episodes(env, 3)

    assert env.cost_limit == pytest.approx(40.0), 'Expected 100 - 3 * 20.'
    assert env.window_cost_mean is None, 'The window clears on each advance.'


def test_cost_gate_blocks_when_over_budget():
    """A satisfied goal gate must not tighten a budget the agent is not meeting."""
    env = _cost_gated_env(cost_per_step=30.0)  # episode cost 150, over 100
    env.reset()

    _run_episodes(env, 3)

    assert env.cost_limit == pytest.approx(100.0), 'The limit must hold while cost exceeds it.'
    assert env.window_cost_mean == pytest.approx(150.0)


def test_cost_gate_is_opt_in():
    """Without the flag the schedule stays open-loop, as before."""
    env = _cost_gated_env(cost_per_step=30.0, gate_on_cost=False)
    env.reset()

    _run_episodes(env, 3)

    assert env.cost_limit == pytest.approx(40.0), 'Cost is ignored unless gate_on_cost is set.'


def test_cost_gate_reopens_when_cost_falls():
    """The gate is a precondition, not a permanent stop."""
    env = _cost_gated_env(cost_per_step=30.0)
    env.reset()

    _run_episodes(env, 2)
    assert env.cost_limit == pytest.approx(100.0)

    env.env.cost_per_step = 1.0  # the multiplier has done its work
    _run_episodes(env, 2)

    assert env.cost_limit == pytest.approx(60.0), 'Expected two advances once cost fits.'


def test_cost_gate_state_round_trip():
    """Cost-window state survives checkpointing."""
    source = _cost_gated_env(cost_per_step=30.0, window=2)
    source.reset()
    _run_episodes(source, 1)

    restored = _cost_gated_env(cost_per_step=30.0, window=2)
    restored.set_state(source.get_state())

    assert restored.window_cost_mean == pytest.approx(source.window_cost_mean)
    assert restored.get_state() == source.get_state()


class _ScriptedEnv(gymnasium.Env):
    """Env whose episodes follow a scripted (goals, cost) cycle, for exact gate tests."""

    EPISODE_LEN = 10

    def __init__(self, script):
        self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), dtype=np.float64)
        self.script = script
        self._episode = 0
        self._step = 0

    def reset(self, **kwargs):  # pylint: disable=unused-argument
        self._step = 0
        return np.zeros(1), {}

    def step(self, action):  # pylint: disable=unused-argument
        goals, cost = self.script[self._episode % len(self.script)]
        self._step += 1
        done = self._step >= self.EPISODE_LEN
        # All the episode's cost lands on its first step; goals on its first `goals` steps.
        info = {'goal_met': self._step <= goals}
        emitted = float(cost) if self._step == 1 else 0.0
        if done:
            self._episode += 1
        return np.zeros(1), 0.0, emitted, False, done, info


def _scripted_curriculum(script, **kwargs):
    params = {
        'initial_cost_limit': 100.0,
        'min_cost_limit': 10.0,
        'decrement': 20.0,
        'success_threshold': 3.0,
        'window': 10,
        'gate_on_cost': False,
    }
    params.update(kwargs)
    return SafeCostLimitCurriculum(_ScriptedEnv(script), **params)


def test_quantile_args_are_validated():
    """A quantile outside [0, 1] is a configuration error."""
    with pytest.raises(ValueError, match='success_quantile must lie in'):
        _scripted_curriculum([(3, 0)], success_quantile=1.5)
    with pytest.raises(ValueError, match='cost_quantile must lie in'):
        _scripted_curriculum([(3, 0)], cost_quantile=-0.1)


# Four episodes of 10 goals and six of none: mean 4.0 clears a threshold of 3.0 although
# 60% of the episodes achieved nothing. The median of the same window is 0.
_BIMODAL_GOALS = [(10, 0), (0, 0), (0, 0)]


def test_mean_goal_gate_is_fooled_by_a_bimodal_window():
    """The behaviour the quantile gate exists to fix, pinned so it cannot regress silently."""
    env = _scripted_curriculum(_BIMODAL_GOALS)
    env.reset()

    _run_episodes(env, 10)

    assert env.cost_limit == pytest.approx(80.0), 'The mean gate advances on a streak.'


def test_median_goal_gate_rejects_a_bimodal_window():
    """Sixty percent of episodes achieving nothing must not count as sustained success."""
    env = _scripted_curriculum(_BIMODAL_GOALS, success_quantile=0.5)
    env.reset()

    _run_episodes(env, 10)

    assert env.cost_limit == pytest.approx(100.0), 'Median 0 is under the threshold of 3.'
    assert env.window_mean == pytest.approx(4.0), 'The mean gate would have cleared this window.'


def test_median_goal_gate_accepts_a_consistent_window():
    """Consistently clearing the bar still advances."""
    env = _scripted_curriculum([(4, 0)], success_quantile=0.5)
    env.reset()

    _run_episodes(env, 10)

    assert env.cost_limit == pytest.approx(80.0)


def test_cost_quantile_rejects_a_bimodal_cost_window():
    """A policy alternating idle and expensive episodes must not pass the cost gate."""
    # Costs alternate 0 and 60: mean 30 <= limit 100 passes, but p75 = 60.
    env = _scripted_curriculum(
        [(4, 0), (4, 60)],
        initial_cost_limit=40.0,
        gate_on_cost=True,
        cost_quantile=0.75,
    )
    env.reset()

    _run_episodes(env, 10)

    assert env.window_cost_mean == pytest.approx(30.0), 'The mean would clear a limit of 40.'
    assert env.cost_limit == pytest.approx(40.0), 'p75 of the window is 60, over the limit.'


def test_cost_quantile_accepts_a_consistent_cost_window():
    """Consistently fitting the budget advances even under the strict quantile."""
    env = _scripted_curriculum(
        [(4, 30)],
        initial_cost_limit=40.0,
        gate_on_cost=True,
        cost_quantile=0.75,
    )
    env.reset()

    _run_episodes(env, 10)

    assert env.cost_limit == pytest.approx(20.0)


def test_summarize_matches_numpy_quantile():
    """The gate statistic is a plain quantile; None keeps the legacy mean."""
    values = [0, 1, 4, 9, 10]
    assert SafeCostLimitCurriculum._summarize(values, None) == pytest.approx(4.8)
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert SafeCostLimitCurriculum._summarize(values, q) == pytest.approx(
            float(np.quantile(np.asarray(values, dtype=float), q)),
        )


def test_cost_limit_survives_action_repeat():
    """Placed inside the repeat, the limit still reaches the outer info dict."""
    env = SafeActionRepeat(_curriculum_env(max_episode_steps=1000), 4)
    env.reset(seed=0)

    _, _, _, _, _, info = env.step(env.action_space.sample())

    assert info['cost_limit'] == pytest.approx(100.0)


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
