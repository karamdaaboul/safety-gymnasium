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
"""Wrapper maintaining a success-gated curriculum over the cost limit."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium
import numpy as np


class SafeCostLimitCurriculum(gymnasium.Wrapper, gymnasium.utils.RecordConstructorArgs):
    """Anneal a cost limit from loose to tight as the agent becomes successful.

    A safe-RL curriculum starts with a loose constraint budget so the agent can learn
    the task at all, then tightens the budget as competence grows. This wrapper tracks
    goals reached per episode (``info['goal_met']``) over a sliding window of the last
    ``window`` *completed* episodes. Whenever the window is full and its mean reaches
    ``success_threshold``, the limit steps down by ``decrement`` -- never below
    ``min_cost_limit`` -- and the window is cleared, so each stage is judged only on
    episodes played under its own limit.

    ``gate_on_cost`` additionally requires the window's episode cost to be at or under the
    *current* limit. Without it the schedule is open-loop: a competent agent keeps the goal
    gate permanently satisfied, so the limit steps down on a timer and slides past the cost
    the agent can actually achieve.

    Both gates compare a mean by default, which a bimodal policy defeats --
    ``[10, 0, 10, 0, 10, 0, 0, 0, 0, 0]`` averages 3.0 and clears a threshold of 3.0 with
    seven empty episodes. ``success_quantile`` and ``cost_quantile`` swap in an order
    statistic. They run in opposite directions because the tail that matters differs:
    goals must be high, so a *low* quantile is strict; cost must be low, so a *high* one is.

    The wrapper is *expose-only*: observations, rewards, costs and the episode flags
    pass through untouched. The current limit is published as ``info['cost_limit']`` on
    every :meth:`reset` and :meth:`step`, and as the :attr:`cost_limit` property, for
    the training loop to feed its constraint (for example a Lagrangian budget).

    Calling :meth:`reset` mid-episode discards the partial episode's goal count rather
    than recording it; the limit and the window survive across resets, so the curriculum
    is scoped to the training run, not to an episode.

    Note:
        Place this wrapper directly around the environment returned by
        :func:`safety_gymnasium.make`, **inside**
        :class:`~safety_gymnasium.wrappers.SafeActionRepeat`, so it observes every
        simulator step's ``goal_met``. It also works outside
        :class:`~safety_gymnasium.wrappers.SafeAutoResetWrapper`, by reading the
        terminal step's ``goal_met`` from ``info['final_info']``.

    Examples:
        >>> env = SafeCostLimitCurriculum(
        ...     safety_gymnasium.make('SafetyPointGoal1-v0'),
        ...     initial_cost_limit=100.0,
        ...     min_cost_limit=25.0,
        ...     decrement=15.0,
        ...     success_threshold=1.5,
        ...     window=20,
        ... )
        >>> obs, info = env.reset(seed=0)
        >>> info['cost_limit']
        100.0
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env: gymnasium.Env,
        initial_cost_limit: float,
        min_cost_limit: float,
        decrement: float,
        success_threshold: float,
        window: int,
        gate_on_cost: bool = False,
        success_quantile: float | None = None,
        cost_quantile: float | None = None,
    ) -> None:
        """Initialize an instance of :class:`SafeCostLimitCurriculum`.

        Args:
            env (gymnasium.Env): The environment to apply the wrapper to.
            initial_cost_limit (float): Cost limit the curriculum starts from.
            min_cost_limit (float): Floor the cost limit never decays below.
            decrement (float): Amount the cost limit steps down by on each advance.
            success_threshold (float): Mean goals per episode over the window that
                triggers an advance.
            window (int): Number of completed episodes the mean is taken over.
            gate_on_cost (bool): Also require the window's episode cost to be at or under
                the current limit before advancing.
            success_quantile (float | None): Compare this quantile of the goal window
                against ``success_threshold`` instead of its mean. Lower is stricter;
                ``0.5`` is the median. ``None`` keeps the mean.
            cost_quantile (float | None): Compare this quantile of the cost window against
                the current limit instead of its mean. Higher is stricter; ``0.5`` is the
                median. ``None`` keeps the mean.
        """
        if min_cost_limit > initial_cost_limit:
            raise ValueError(
                f'min_cost_limit must not exceed initial_cost_limit, '
                f'got {min_cost_limit} > {initial_cost_limit}.',
            )
        if decrement <= 0:
            raise ValueError(f'decrement must be positive, got {decrement}.')
        if success_threshold < 0:
            raise ValueError(f'success_threshold must be non-negative, got {success_threshold}.')
        if window < 1:
            raise ValueError(f'window must be a positive integer, got {window}.')
        for name, quantile in (
            ('success_quantile', success_quantile),
            ('cost_quantile', cost_quantile),
        ):
            if quantile is not None and not 0.0 <= quantile <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1] or be None, got {quantile}.')

        gymnasium.utils.RecordConstructorArgs.__init__(
            self,
            initial_cost_limit=initial_cost_limit,
            min_cost_limit=min_cost_limit,
            decrement=decrement,
            success_threshold=success_threshold,
            window=window,
            gate_on_cost=gate_on_cost,
            success_quantile=success_quantile,
            cost_quantile=cost_quantile,
        )
        gymnasium.Wrapper.__init__(self, env)

        self.min_cost_limit = min_cost_limit
        self.decrement = decrement
        self.success_threshold = success_threshold
        self.window = window
        self.gate_on_cost = gate_on_cost
        self.success_quantile = success_quantile
        self.cost_quantile = cost_quantile
        self._cost_limit = float(initial_cost_limit)
        self._episode_goals = 0
        self._episode_cost = 0.0
        self._recent_goals: deque[int] = deque(maxlen=window)
        self._recent_costs: deque[float] = deque(maxlen=window)

    @property
    def cost_limit(self) -> float:
        """The current cost limit, for the training loop's constraint."""
        return self._cost_limit

    @property
    def episode_goals(self) -> int:
        """Goals reached so far in the current episode."""
        return self._episode_goals

    @property
    def window_mean(self) -> float | None:
        """Mean goals per episode over the window, or ``None`` while it is empty."""
        if not self._recent_goals:
            return None
        return sum(self._recent_goals) / len(self._recent_goals)

    @property
    def window_cost_mean(self) -> float | None:
        """Mean cost per episode over the window, or ``None`` while it is empty."""
        if not self._recent_costs:
            return None
        return sum(self._recent_costs) / len(self._recent_costs)

    def get_state(self) -> dict[str, Any]:
        """Return the curriculum state for checkpointing."""
        return {
            'cost_limit': self._cost_limit,
            'episode_goals': self._episode_goals,
            'episode_cost': self._episode_cost,
            'recent_goals': list(self._recent_goals),
            'recent_costs': list(self._recent_costs),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore curriculum state produced by :meth:`get_state`."""
        self._cost_limit = float(state['cost_limit'])
        self._episode_goals = int(state['episode_goals'])
        self._episode_cost = float(state.get('episode_cost', 0.0))
        self._recent_goals = deque(state['recent_goals'], maxlen=self.window)
        self._recent_costs = deque(state.get('recent_costs', []), maxlen=self.window)

    def reset(self, **kwargs):
        """Reset the environment, discarding any partial episode's goal count."""
        obs, info = self.env.reset(**kwargs)
        # A partial episode would bias the window means down, so it is discarded rather
        # than recorded; the limit and the window survive across resets.
        self._episode_goals = 0
        self._episode_cost = 0.0
        info['cost_limit'] = self._cost_limit
        return obs, info

    def step(self, action):
        """Step the environment, advancing the curriculum on episode boundaries."""
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        # An autoreset wrapper inside this one moves the terminal step's info under
        # `final_info`; read `goal_met` from wherever it actually is. `cost` is returned
        # positionally and is the terminal step's own cost either way.
        step_info = info.get('final_info', info)
        if step_info.get('goal_met', False):
            self._episode_goals += 1
        self._episode_cost += float(cost)

        if terminated or truncated:
            self._recent_goals.append(self._episode_goals)
            self._recent_costs.append(self._episode_cost)
            self._episode_goals = 0
            self._episode_cost = 0.0
            if self._cost_limit > self.min_cost_limit and self._advance_gate_open():
                self._cost_limit = max(self._cost_limit - self.decrement, self.min_cost_limit)
                # Judge the next stage only on episodes played under the new limit.
                self._recent_goals.clear()
                self._recent_costs.clear()

        info['cost_limit'] = self._cost_limit
        return obs, reward, cost, terminated, truncated, info

    def _advance_gate_open(self) -> bool:
        """Whether a full window justifies tightening the limit."""
        if len(self._recent_goals) < self.window:
            return False
        if self._summarize(self._recent_goals, self.success_quantile) < self.success_threshold:
            return False
        if not self.gate_on_cost:
            return True
        return self._summarize(self._recent_costs, self.cost_quantile) <= self._cost_limit

    @staticmethod
    def _summarize(values, quantile: float | None) -> float:
        """The window statistic a gate compares against: its mean, or a quantile of it."""
        if quantile is None:
            return sum(values) / len(values)
        return float(np.quantile(np.asarray(values, dtype=float), quantile))
