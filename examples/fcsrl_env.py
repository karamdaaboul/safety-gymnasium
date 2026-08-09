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
"""Example of the FCSRL-style wrapper stack.

FCSRL (https://github.com/czp16/FCSRL) trains on safety-gymnasium with three env-side
modifications that its paper does not mention: an action repeat of 4 on both training and
evaluation environments, goal achievement treated as a pseudo-terminal for value
bootstrapping, and a running mean/std normalization of observations. This script builds
that stack and reports the agent-step vs simulator-step accounting it implies.

Run with::

    MUJOCO_GL=egl python examples/fcsrl_env.py
    MUJOCO_GL=egl python examples/fcsrl_env.py --pixels
"""

import argparse

import safety_gymnasium
from safety_gymnasium.wrappers import (
    SafeActionRepeat,
    SafeGoalMetTerminal,
    SafeNormalizeObservation,
    SafePixelObservation,
)


def build_env(env_name, action_repeat, pixels):
    """Assemble the wrapper stack.

    Order matters. ``safety_gymnasium.make`` already applies ``SafeTimeLimit``, so the
    time limit sits *inside* the action repeat and therefore counts simulator steps: a
    1000 step environment becomes a 250 agent step episode covering the same 1000
    simulator steps, and episode return and cost stay comparable to the unrepeated
    benchmark.

    ``SafePixelObservation`` goes *outside* the repeat so the camera is rendered once per
    agent step instead of once per simulator step.
    """
    env = safety_gymnasium.make(env_name)
    env = SafeGoalMetTerminal(env)
    env = SafeActionRepeat(env, action_repeat)
    if pixels:
        return SafePixelObservation(env, size=(64, 64))
    return SafeNormalizeObservation(env)


def run_episode(env):
    """Run one episode with random actions and collect the step statistics."""
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), 'Reset observation outside the space.'

    agent_steps, sim_steps, goals_met = 0, 0, 0
    ep_ret, ep_cost = 0.0, 0.0

    while True:
        obs, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())

        agent_steps += 1
        sim_steps += info['sim_steps']
        goals_met += int(info.get('goal_met', False))
        ep_ret += reward
        ep_cost += cost

        if terminated or truncated:
            break

    return agent_steps, sim_steps, goals_met, ep_ret, ep_cost


def main(env_name, action_repeat, pixels):
    """Build the stack, run an episode and print the step accounting."""
    env = build_env(env_name, action_repeat, pixels)
    agent_steps, sim_steps, goals_met, ep_ret, ep_cost = run_episode(env)

    print(f'Environment:     {env_name}')
    print(f'Observation:     {env.observation_space}')
    print(f'Action repeat:   {action_repeat}')
    print(f'Episode length:  {agent_steps} agent steps / {sim_steps} simulator steps')
    print(f'Episode return:  {ep_ret:.3f}')
    print(f'Episode cost:    {ep_cost:.1f}')
    print(f'Goals reached:   {goals_met}')
    print()
    print(
        'A run reported as N agent steps consumes '
        f'{action_repeat}N simulator steps. FCSRL reports 2M steps, which is '
        '8M simulator steps.',
    )

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SafetyPointGoal1-v0')
    parser.add_argument('--action-repeat', type=int, default=4)
    parser.add_argument('--pixels', action='store_true', help='use 64x64 image observations')
    args = parser.parse_args()
    main(args.env, args.action_repeat, args.pixels)
