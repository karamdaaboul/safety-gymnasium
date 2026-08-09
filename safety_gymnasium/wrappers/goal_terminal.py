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
"""Wrapper exposing goal achievement as a pseudo-terminal for value bootstrapping."""

from __future__ import annotations

import gymnasium


class SafeGoalMetTerminal(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    r"""Report goal achievement as a pseudo-terminal in ``info['pseudo_terminated']``.

    Navigation tasks respawn the goal in place when it is reached
    (``mechanism_conf.continue_goal`` is ``True`` by default), so
    :meth:`safety_gymnasium.builder.Builder.step` sets ``info['goal_met']`` but leaves
    ``terminated`` as ``False`` and the episode running. That is correct as an episode
    boundary, but it is misleading as a *value* boundary: the state reached at goal
    achievement is followed by a fresh, unrelated goal, so bootstrapping the value of the
    successor state across that discontinuity is unsound.

    This wrapper leaves the episode alone -- ``terminated`` and ``truncated`` are passed
    through untouched and still drive resets and episode logging -- and only adds a flag
    that a learner can use to cut the bootstrap:

    .. math::

        V(s) = r + \gamma \, (1 - \mathrm{pseudo\_terminated}) \, V(s')

    The intended split is that ``pseudo_terminated`` zeroes the bootstrap and cuts an
    n-step chain, while ``truncated`` cuts the chain but *keeps* the bootstrap.

    Note:
        ``pseudo_terminated`` is ``terminated or goal_met``, so genuine terminations
        (agent death, goal resampling failure) also cut the bootstrap. The reference
        implementation this reproduces sets the flag from ``goal_met`` alone, which
        silently bootstraps through real terminal states; that only goes unnoticed
        because the agents it was run with never die.

    Examples:
        >>> env = SafeGoalMetTerminal(safety_gymnasium.make('SafetyPointGoal1-v0'))
        >>> obs, reward, cost, terminated, truncated, info = env.step(action)
        >>> info['pseudo_terminated']  # True on the step that reaches the goal
        False
    """

    def __init__(self, env: gymnasium.Env) -> None:
        """Initialize an instance of :class:`SafeGoalMetTerminal`."""
        gymnasium.utils.RecordConstructorArgs.__init__(self)
        gymnasium.Wrapper.__init__(self, env)

    def step(self, action):
        """Step the environment and annotate ``info`` with the pseudo-terminal flag."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info['pseudo_terminated'] = bool(terminated) or bool(info.get('goal_met', False))
        return obs, reward, cost, terminated, truncated, info
