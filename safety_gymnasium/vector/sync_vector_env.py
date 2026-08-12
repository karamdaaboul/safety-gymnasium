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
"""The sync vectorized environment."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.utils import concatenate, iterate
from gymnasium.vector.vector_env import ArrayType, AutoresetMode

from safety_gymnasium.vector.utils.tile_images import tile_images


__all__ = ['SafetySyncVectorEnv']


class SafetySyncVectorEnv(SyncVectorEnv):
    """Vectored safe environment that serially runs multiple safe environments.

    This adds the ``cost`` channel to Gymnasium's :class:`SyncVectorEnv`, so
    :meth:`step` returns ``(obs, rewards, costs, terminateds, truncateds, infos)``.

    Note:
        Sub-environments are auto-reset within the same step that ends an episode, matching
        :class:`~safety_gymnasium.vector.SafetyAsyncVectorEnv`. The final transition is
        available as ``infos['final_observation']`` and ``infos['final_info']``; the
        returned observation is already the first one of the next episode. Every other
        ``info`` key on that step therefore comes from :meth:`reset`, not from the step
        that ended the episode.
    """

    def __init__(
        self,
        env_fns: Iterator[Callable[[], Env]],
        copy: bool = True,
    ) -> None:
        """Initializes the vectorized safe environment."""
        super().__init__(env_fns, copy, autoreset_mode=AutoresetMode.SAME_STEP)
        self._costs = np.zeros((self.num_envs,), dtype=np.float64)

    def render(self) -> np.ndarray:
        """Render the environment."""
        # get the images.
        imgs = self.get_images()
        # tile the images.
        return tile_images(imgs)

    def step(
        self,
        actions: ActType,
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through each of the environments returning the batched results.

        This overrides :meth:`SyncVectorEnv.step` rather than ``step_wait``: Gymnasium
        dropped the two-phase ``step_async``/``step_wait`` split for the synchronous
        vector environment, and its :meth:`step` unpacks a five-tuple, which cannot carry
        the cost.
        """
        actions = iterate(self.action_space, actions)

        infos = {}
        # `strict=True` mirrors Gymnasium: a mis-sized action batch must raise rather
        # than silently leave the trailing sub-environments unstepped and their
        # rewards, costs and flags stale from the previous step.
        for i, (action, _) in enumerate(zip(actions, self.envs, strict=True)):
            (
                self._env_obs[i],
                self._rewards[i],
                self._costs[i],
                self._terminations[i],
                self._truncations[i],
                env_info,
            ) = self.envs[i].step(action)

            if self._terminations[i] or self._truncations[i]:
                final_obs, final_info = self._env_obs[i], env_info
                self._env_obs[i], env_info = self.envs[i].reset()
                env_info['final_observation'] = final_obs
                env_info['final_info'] = final_info
                # Gymnasium's SAME_STEP consumers (RecordEpisodeStatistics, the
                # vector observation wrappers) look for `final_obs`, and `_add_info`
                # special-cases that key. Emit it alongside the `final_observation`
                # name this package has always used.
                env_info['final_obs'] = final_obs

            infos = self._add_info(infos, env_info, i)

        self._observations = concatenate(
            self.single_observation_space,
            self._env_obs,
            self._observations,
        )

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._costs),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def get_images(self) -> list[np.ndarray]:
        """Get images from child environments.

        The sub-environments must have been created with ``render_mode='rgb_array'``.
        """
        return [env.render() for env in self.envs]
