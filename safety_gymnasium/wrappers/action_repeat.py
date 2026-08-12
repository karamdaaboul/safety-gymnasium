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
"""Wrapper for repeating each action over several simulator steps."""

from __future__ import annotations

from typing import Any

import gymnasium


#: Info keys that are aggregated by logical ``or`` rather than by summing.
BOOLEAN_INFO_KEYS = ('goal_met', 'pseudo_terminated')

#: ``cost_*``-prefixed keys that are levels rather than per-step increments, so they take
#: the overwrite rule instead of being summed.
LEVEL_INFO_KEYS = ('cost_limit',)


def merge_info(accumulator: dict[str, Any], step_info: dict[str, Any]) -> None:
    """Merge one simulator step's ``info`` into an accumulator, in place.

    The rules mirror how the other step outputs are aggregated by
    :class:`SafeActionRepeat`:

    - ``cost_*`` entries (``cost_sum``, ``cost_hazards``, ``cost_exception``, ...) are
      **summed**, exactly like the returned ``cost``. Each is an indicator in
      :math:`\\{0, 1\\}` per simulator step, so summing keeps an episode's cost total equal
      to what the unrepeated environment would report. :data:`LEVEL_INFO_KEYS` are exempt:
      ``cost_limit`` is a level, not an increment, and summing it would scale it by the
      repeat.
    - :data:`BOOLEAN_INFO_KEYS` are combined with logical ``or``.
    - Anything else is overwritten by the most recent simulator step, matching the
      returned observation.
    """
    for key, value in step_info.items():
        if key.startswith('cost_') and key not in LEVEL_INFO_KEYS:
            accumulator[key] = accumulator.get(key, 0.0) + value
        elif key in BOOLEAN_INFO_KEYS:
            accumulator[key] = bool(accumulator.get(key, False)) or bool(value)
        else:
            accumulator[key] = value


class SafeActionRepeat(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    """Apply each action for ``n_repeat`` consecutive simulator steps.

    One :meth:`step` of this wrapper is one *agent* step and up to ``n_repeat``
    *simulator* steps. Rewards and costs are summed over the repeat, the last
    observation is returned, and ``terminated``/``truncated`` are combined with logical
    ``or``, breaking out of the loop as soon as either becomes ``True``.

    Because summing preserves totals, an episode's return and cost are unchanged by this
    wrapper as long as the time limit stays *inside* it -- which is the case for any
    environment built by :func:`safety_gymnasium.make`, since that applies
    :class:`~safety_gymnasium.wrappers.SafeTimeLimit` itself. A 1000 step environment
    therefore becomes a 250 agent step episode covering the same 1000 simulator steps.

    ``info['sim_steps']`` reports how many simulator steps the agent step actually
    consumed, so that a training loop can log simulator steps alongside agent steps
    instead of conflating the two.

    Examples:
        >>> env = SafeActionRepeat(safety_gymnasium.make('SafetyPointGoal1-v0'), 4)
        >>> obs, info = env.reset(seed=0)
        >>> obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
        >>> info['sim_steps']
        4
    """

    def __init__(self, env: gymnasium.Env, n_repeat: int) -> None:
        """Initialize an instance of :class:`SafeActionRepeat`.

        Args:
            env (gymnasium.Env): The environment to apply the wrapper to.
            n_repeat (int): Number of simulator steps each action is applied for.
        """
        if n_repeat < 1:
            raise ValueError(f'n_repeat must be a positive integer, got {n_repeat}.')

        gymnasium.utils.RecordConstructorArgs.__init__(self, n_repeat=n_repeat)
        gymnasium.Wrapper.__init__(self, env)

        self.n_repeat = n_repeat

    def step(self, action):
        """Step the underlying environment ``n_repeat`` times with the same action."""
        total_reward = 0.0
        total_cost = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        sim_steps = 0
        obs = None

        for _ in range(self.n_repeat):
            obs, reward, cost, step_terminated, step_truncated, step_info = self.env.step(action)
            # Count the wrapped environment's own simulator steps if it reports them, so
            # that nesting this wrapper composes instead of undercounting.
            sim_steps += step_info.get('sim_steps', 1)
            total_reward += reward
            total_cost += cost
            terminated = terminated or step_terminated
            truncated = truncated or step_truncated
            merge_info(info, step_info)

            if step_terminated or step_truncated:
                break

        info['sim_steps'] = sim_steps

        return obs, total_reward, total_cost, terminated, truncated, info
