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
"""Test the vectorized environments."""

import numpy as np
import pytest  # pylint: disable=import-error

import helpers
import safety_gymnasium
from safety_gymnasium.wrappers import SafeActionRepeat, SafeGoalMetTerminal


ENV_ID = 'SafetyPointGoal1-v0'


@helpers.parametrize(asynchronous=[False, True])
def test_vector_step_returns_cost(asynchronous):
    """Both vector environments carry the cost channel through ``step``."""
    env = safety_gymnasium.vector.make(ENV_ID, num_envs=3, asynchronous=asynchronous)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 60)

        outputs = env.step(env.action_space.sample())
        assert len(outputs) == 6, 'Expected (obs, reward, cost, terminated, truncated, info).'

        obs, reward, cost, terminated, truncated, _ = outputs
        assert obs.shape == (3, 60)
        for batched in (reward, cost, terminated, truncated):
            assert batched.shape == (3,)
        assert np.all(np.isfinite(cost))
    finally:
        env.close()


@helpers.parametrize(asynchronous=[False, True])
def test_vector_matches_individual_envs(asynchronous):
    """A vector env reproduces the trajectories of separately seeded single envs."""
    num_envs, num_steps = 2, 20
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, size=(num_steps, num_envs, 2))

    env = safety_gymnasium.vector.make(ENV_ID, num_envs=num_envs, asynchronous=asynchronous)
    try:
        env.reset(seed=0)
        batched = [env.step(np.array(action))[:3] for action in actions]
    finally:
        env.close()

    # `reset(seed=0)` seeds the sub-environments with 0, 1, ..., num_envs - 1.
    for index in range(num_envs):
        single = safety_gymnasium.make(ENV_ID)
        single.reset(seed=index)
        for step, action in enumerate(actions):
            obs, reward, cost, _, _, _ = single.step(action[index])
            assert obs == pytest.approx(batched[step][0][index])
            assert reward == pytest.approx(batched[step][1][index])
            assert cost == pytest.approx(batched[step][2][index])


def test_sync_and_async_agree():
    """The synchronous and asynchronous implementations are interchangeable."""
    actions = np.random.default_rng(1).uniform(-1.0, 1.0, size=(20, 2, 2))

    rollouts = {}
    for asynchronous in (False, True):
        env = safety_gymnasium.vector.make(ENV_ID, num_envs=2, asynchronous=asynchronous)
        try:
            env.reset(seed=3)
            rollouts[asynchronous] = [
                tuple(np.copy(item) for item in env.step(np.array(action))[:3])
                for action in actions
            ]
        finally:
            env.close()

    for sync_step, async_step in zip(rollouts[False], rollouts[True], strict=True):
        for sync_value, async_value in zip(sync_step, async_step, strict=True):
            assert sync_value == pytest.approx(async_value)


@helpers.parametrize(asynchronous=[False, True])
def test_vector_autoresets_within_the_step(asynchronous):
    """An ended episode is reset in the same step, exposing the final transition."""
    env = safety_gymnasium.vector.make(
        ENV_ID,
        num_envs=2,
        asynchronous=asynchronous,
        max_episode_steps=5,
    )
    try:
        env.reset(seed=0)
        for _ in range(5):
            _, _, _, _, truncated, info = env.step(env.action_space.sample())

        assert np.all(truncated)
        assert 'final_observation' in info
        assert 'final_info' in info
    finally:
        env.close()


@helpers.parametrize(asynchronous=[False, True])
def test_vector_applies_wrappers_to_each_env(asynchronous):
    """Per-environment wrappers survive vectorization."""

    def wrap(env):
        return SafeActionRepeat(SafeGoalMetTerminal(env), 4)

    env = safety_gymnasium.vector.make(
        ENV_ID,
        num_envs=2,
        asynchronous=asynchronous,
        wrappers=wrap,
    )
    try:
        env.reset(seed=0)
        _, _, _, _, _, info = env.step(env.action_space.sample())
        assert np.all(info['sim_steps'] == 4)
    finally:
        env.close()


def test_sync_render_tiles_frames():
    """``render`` tiles the sub-environment frames into a single image."""
    env = safety_gymnasium.vector.make(
        ENV_ID,
        num_envs=2,
        asynchronous=False,
        render_mode='rgb_array',
    )
    try:
        env.reset(seed=0)
        env.step(env.action_space.sample())
        image = env.render()
    finally:
        env.close()

    assert image.ndim == 3
    assert image.shape[-1] == 3
