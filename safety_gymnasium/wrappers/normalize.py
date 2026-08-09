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
"""Wrapper for normalizing the output of an environment."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from safety_gymnasium.utils.normalizer import MeanStdNormalizer, RunningMeanStd


class SafeNormalizeObservation(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    """Normalize observations by a running mean and standard deviation.

    The running statistics are updated from every observation the wrapper returns, and
    normalized values are clipped to ``[-clip, clip]``.

    Evaluation must not move the statistics, so call :meth:`freeze` before evaluating and
    :meth:`unfreeze` afterwards. To keep a training and an evaluation environment in sync,
    build one and hand its :attr:`normalizer` to the other::

        train_env = SafeNormalizeObservation(safety_gymnasium.make(env_id))
        eval_env = SafeNormalizeObservation(safety_gymnasium.make(env_id))
        eval_env.normalizer = train_env.normalizer
        eval_env.freeze()

    Use :meth:`get_stats` and :meth:`set_stats` to persist the statistics alongside a
    policy checkpoint; a policy restored without them sees a different input distribution
    than it was trained on.

    Note:
        Only ``Box`` observation spaces are supported. For pixel observations, whose
        per-channel statistics are not meaningful in the same way, see
        :class:`~safety_gymnasium.wrappers.SafePixelObservation` instead.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        clip: float = 50.0,
        epsilon: float = 1e-20,
    ) -> None:
        """Initialize an instance of :class:`SafeNormalizeObservation`.

        Args:
            env (gymnasium.Env): The environment to apply the wrapper to.
            clip (float): Normalized observations are clipped to ``[-clip, clip]``.
            epsilon (float): Stability term added to the variance before taking the
                square root.
        """
        gymnasium.utils.RecordConstructorArgs.__init__(self, clip=clip, epsilon=epsilon)
        gymnasium.Wrapper.__init__(self, env)

        if not isinstance(self.observation_space, gymnasium.spaces.Box):
            raise TypeError(
                'SafeNormalizeObservation requires a Box observation space, got '
                f'{type(self.observation_space).__name__}.',
            )

        self.normalizer = MeanStdNormalizer(clip=clip, epsilon=epsilon, read_only=False)
        self.observation_space = gymnasium.spaces.Box(
            -clip,
            clip,
            self.observation_space.shape,
            dtype=self.observation_space.dtype,
        )

    def freeze(self) -> None:
        """Stop updating the running statistics, e.g. while evaluating."""
        self.normalizer.set_read_only()

    def unfreeze(self) -> None:
        """Resume updating the running statistics."""
        self.normalizer.unset_read_only()

    def get_stats(self) -> dict[str, Any] | None:
        """Return the running statistics for checkpointing."""
        return self.normalizer.state_dict()

    def set_stats(self, state: dict[str, Any] | None) -> None:
        """Restore running statistics produced by :meth:`get_stats`."""
        self.normalizer.load_state_dict(state)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a single observation, updating the statistics unless frozen."""
        return self.normalizer(np.expand_dims(obs, axis=0))[0].astype(
            self.observation_space.dtype,
        )

    def reset(self, **kwargs):
        """Reset the environment and normalize the initial observation."""
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def step(self, action):
        """Step the environment and normalize the observation."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        if 'final_observation' in info:
            info['original_final_observation'] = info['final_observation']
            info['final_observation'] = self.normalize(info['final_observation'])
        return self.normalize(obs), reward, cost, terminated, truncated, info


class SafeNormalizeReward(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    r"""Scale rewards so that the exponential moving average of the return has fixed variance.

    The moving average has variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories, so rewards will not be scaled correctly
        if the wrapper was newly instantiated or the policy changed recently.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize an instance of :class:`SafeNormalizeReward`.

        Args:
            env (gymnasium.Env): The environment to apply the wrapper to.
            gamma (float): Discount factor used in the exponential moving average.
            epsilon (float): A stability parameter.
        """
        gymnasium.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gymnasium.Wrapper.__init__(self, env)

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Step the environment, normalizing the reward returned."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self.returns = self.returns * self.gamma * (1 - float(terminated)) + reward
        reward = self.normalize(np.array([reward]))[0]
        return obs, reward, cost, terminated, truncated, info

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards by the standard deviation of the running returns."""
        self.return_rms.update(self.returns)
        return rewards / np.sqrt(self.return_rms.var + self.epsilon)


class SafeNormalizeCost(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    r"""Scale costs so that the exponential moving average of the cost-return has fixed variance.

    The moving average has variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories, so costs will not be scaled correctly if
        the wrapper was newly instantiated or the policy changed recently.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize an instance of :class:`SafeNormalizeCost`.

        Args:
            env (gymnasium.Env): The environment to apply the wrapper to.
            gamma (float): Discount factor used in the exponential moving average.
            epsilon (float): A stability parameter.
        """
        gymnasium.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gymnasium.Wrapper.__init__(self, env)

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(1)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Step the environment, normalizing the cost returned."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self.returns = self.returns * self.gamma * (1 - float(terminated)) + cost
        cost = self.normalize(np.array([cost]))[0]
        return obs, reward, cost, terminated, truncated, info

    def normalize(self, costs: np.ndarray) -> np.ndarray:
        """Normalize costs by the standard deviation of the running cost-returns."""
        self.return_rms.update(self.returns)
        return costs / np.sqrt(self.return_rms.var + self.epsilon)
