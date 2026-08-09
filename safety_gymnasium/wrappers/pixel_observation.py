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
"""Wrapper replacing the observation with a rendering of the agent's vision camera."""

from __future__ import annotations

import gymnasium
import numpy as np


class SafePixelObservation(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    """Replace the observation with an RGB rendering of the agent's ``vision`` camera.

    The frame is scaled to ``[0, 1]`` and kept in ``HWC`` layout, giving an observation
    space of ``Box(0.0, 1.0, (height, width, 3), float32)``.

    This is an alternative to the registered ``Safety*Vision-v0`` environment ids. Those
    set ``observe_vision=True`` on the task, which renders the camera inside every call to
    :meth:`safety_gymnasium.bases.base_task.BaseTask.obs` -- that is, on every *simulator*
    step. Rendering here instead means the camera is rendered once per call to this
    wrapper's :meth:`step`, so placing it outside
    :class:`~safety_gymnasium.wrappers.SafeActionRepeat` renders one frame per agent step
    rather than one per simulator step::

        env = SafePixelObservation(
            SafeActionRepeat(SafeGoalMetTerminal(safety_gymnasium.make('SafetyPointGoal1-v0')), 4),
            size=(64, 64),
        )

    Note:
        The rendered frame includes the lidar and compass marker geoms that
        :meth:`safety_gymnasium.bases.underlying.Underlying.render` draws, so the pixels
        are not a plain camera view of the scene.

    Note:
        Interleaving this wrapper with :meth:`render` calls at a *different* resolution
        forces the offscreen framebuffer to be recreated on every switch, which is slow.
        Prefer matching the sizes, or rendering separately.
    """

    def __init__(self, env: gymnasium.Env, size: tuple[int, int] = (64, 64)) -> None:
        """Initialize an instance of :class:`SafePixelObservation`.

        Args:
            env (gymnasium.Env): The environment to apply the wrapper to.
            size (tuple): Frame size as ``(height, width)``.
        """
        gymnasium.utils.RecordConstructorArgs.__init__(self, size=size)
        gymnasium.Wrapper.__init__(self, env)

        if not hasattr(env.unwrapped, 'task'):
            raise TypeError(
                'SafePixelObservation requires an environment with a `task`, such as the '
                'navigation and vision environments built on '
                '`safety_gymnasium.builder.Builder`. '
                f'{type(env.unwrapped).__name__} has none.',
            )

        self.height, self.width = size
        self.observation_space = gymnasium.spaces.Box(
            0.0,
            1.0,
            (self.height, self.width, 3),
            dtype=np.float32,
        )

    def observation(self) -> np.ndarray:
        """Render the agent's vision camera and scale it to ``[0, 1]``."""
        frame = self.env.unwrapped.task.render(
            width=self.width,
            height=self.height,
            mode='rgb_array',
            camera_name='vision',
            cost={},
        )
        return np.asarray(frame, dtype=np.float32) / 255.0

    def reset(self, **kwargs):
        """Reset the environment and return the rendered observation."""
        _, info = self.env.reset(**kwargs)
        return self.observation(), info

    def step(self, action):
        """Step the environment and return the rendered observation."""
        _, reward, cost, terminated, truncated, info = self.env.step(action)
        return self.observation(), reward, cost, terminated, truncated, info
