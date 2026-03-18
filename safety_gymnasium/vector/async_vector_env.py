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
"""An async vector environment."""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from copy import deepcopy
from multiprocessing import connection
from typing import Sequence

import gymnasium
import numpy as np
from gymnasium.error import NoAsyncCallError
from gymnasium.vector.async_vector_env import AsyncState, AsyncVectorEnv
from gymnasium.vector.utils import concatenate, write_to_shared_memory

from safety_gymnasium.vector.utils.tile_images import tile_images


__all__ = ['AsyncVectorEnv']


class SafetyAsyncVectorEnv(AsyncVectorEnv):
    """The async vectorized environment for Safety-Gymnasium."""

    def __init__(
        self,
        env_fns: Sequence[callable],
        shared_memory: bool = True,
        copy: bool = True,
        context: str | None = None,
        daemon: bool = True,
        worker: callable | None = None,
    ) -> None:
        """Initialize the async vector environment.

        Args:
            env_fns: A list of callable functions that create the environments.
            shared_memory: Whether to use shared memory for communication.
            copy: Whether to copy the observation.
            context: The context type of multiprocessing.
            daemon: Whether the workers are daemons.
            worker: The worker function.
        """
        target = _worker_shared_memory if shared_memory else _worker
        target = worker or target
        super().__init__(
            env_fns,
            shared_memory,
            copy,
            context,
            daemon,
            worker=target,
        )

    def get_images(self):
        """Get the images from the child environment."""
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def render(self):
        """Render the environment."""
        # get the images.
        imgs = self.get_images()
        # tile the images.
        return tile_images(imgs)

    # pylint: disable-next=too-many-locals
    def step_wait(
        self,
        timeout: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait`
            times out. If ``None``, the call to :meth:`step_wait` never times out.
        """
        # check if the environment is running.
        self._assert_is_running()
        # check if the state is waiting for step.
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                'Calling `step_wait` without any prior call to `step_async`.',
                AsyncState.WAITING_STEP.value,
            )

        # wait for the results.
        poll_fn = getattr(self, '_poll_pipe_envs', None) or getattr(self, '_poll')
        if not poll_fn(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f'The call to `step_wait` has timed out after {timeout} second(s).',
            )

        # get the results.
        observations_list, rewards, costs, terminateds, truncateds, infos = [], [], [], [], [], {}
        successes = []
        for idx, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            obs, rew, cost, terminated, truncated, info = result

            successes.append(success)
            if success:
                observations_list.append(obs)
                rewards.append(rew)
                costs.append(cost)
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos = self._add_info(infos, info, idx)

        # check if there are any errors.
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space,
                observations_list,
                self.observations,
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(costs),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )


def _worker(
    index: int,
    env_fn: callable,
    pipe: connection.Connection,
    parent_pipe: connection.Connection,
    shared_memory: bool,
    error_queue: mp.Queue,
    autoreset_mode=None,
) -> None:
    """The worker function for the async vector environment."""
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation, info = env.reset(**data)
                pipe.send(((observation, info), True))
            elif command == 'reset-noop':
                pipe.send(((None, {}), True))
            elif command == 'step':
                observation, reward, cost, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info['final_observation'] = old_observation
                    info['final_info'] = old_info
                pipe.send(((observation, reward, cost, terminated, truncated, info), True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_call':
                name, args, kwargs = data
                if name in ['reset', 'step', 'close', '_setattr', '_check_spaces']:
                    raise ValueError(
                        f'Trying to call function `{name}` with `call`, use `{name}` directly instead.',
                    )
                attr = env.get_wrapper_attr(name) if hasattr(env, 'get_wrapper_attr') else getattr(env, name)
                if callable(attr):
                    pipe.send((attr(*args, **kwargs), True))
                else:
                    pipe.send((attr, True))
            elif command == '_setattr':
                name, value = data
                if hasattr(env, 'set_wrapper_attr'):
                    env.set_wrapper_attr(name, value)
                else:
                    setattr(env, name, value)
                pipe.send((None, True))
            elif command == '_check_spaces':
                obs_mode, single_obs_space, single_action_space = data
                pipe.send(
                    ((single_obs_space == env.observation_space, single_action_space == env.action_space), True),
                )
            else:
                raise RuntimeError(f'Received unknown command `{command}`.')
    # pylint: disable-next=broad-except
    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()
        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(
    index: int,
    env_fn: callable,
    pipe: connection.Connection,
    parent_pipe: connection.Connection,
    shared_memory: bool,
    error_queue: mp.Queue,
    autoreset_mode=None,
) -> None:
    """The shared memory version of worker function for the async vector environment."""
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation, info = env.reset(**data)
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, info), True))
            elif command == 'reset-noop':
                pipe.send(((None, {}), True))
            elif command == 'step':
                observation, reward, cost, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info['final_observation'] = old_observation
                    info['final_info'] = old_info
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, reward, cost, terminated, truncated, info), True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_call':
                name, args, kwargs = data
                if name in ['reset', 'step', 'close', '_setattr', '_check_spaces']:
                    raise ValueError(
                        f'Trying to call function `{name}` with `call`, use `{name}` directly instead.',
                    )
                attr = env.get_wrapper_attr(name) if hasattr(env, 'get_wrapper_attr') else getattr(env, name)
                if callable(attr):
                    pipe.send((attr(*args, **kwargs), True))
                else:
                    pipe.send((attr, True))
            elif command == '_setattr':
                name, value = data
                if hasattr(env, 'set_wrapper_attr'):
                    env.set_wrapper_attr(name, value)
                else:
                    setattr(env, name, value)
                pipe.send((None, True))
            elif command == '_check_spaces':
                obs_mode, single_obs_space, single_action_space = data
                pipe.send(
                    ((single_obs_space == observation_space, single_action_space == env.action_space), True),
                )
            else:
                raise RuntimeError(f'Received unknown command `{command}`.')
    # pylint: disable-next=broad-except
    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        trace = traceback.format_exc()
        error_queue.put((index, error_type, error_message, trace))
        pipe.send((None, False))
    finally:
        env.close()
