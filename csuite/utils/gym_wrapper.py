# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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

"""Wrapper for adapating a csuite base.Environment to OpenAI gym interface."""

import typing
from typing import Any, Dict, Tuple, Optional, Union

from csuite.environments import base
from dm_env import specs
import gym
from gym import spaces
import numpy as np

# OpenAI gym step format = obs, reward, is_finished, other_info
_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class GymFromCSuite(gym.Env):
  """A wrapper to convert a CSuite environment to an OpenAI gym.Env."""

  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, csuite_env: base.Environment):
    self._csuite_env = csuite_env
    self._np_random = None  # GYM env_checker requires a _np_random attr
    self.viewer = None

  def step(self, action) -> _GymTimestep:
    # Convert the csuite step result to a gym timestep.
    try:
      observation, reward = self._csuite_env.step(action)
    except RuntimeError as e:
      # The gym.utils.env_checker expects the following assertion.
      if str(e) == base.STEP_WITHOUT_START_ERR:
        assert False, 'Cannot call env.step() before calling reset()'

    return observation, reward, False, {}

  def reset(self,
            seed: Optional[int] = None,
            options: Optional[dict] = None):  # pylint: disable=g-bare-generic
    if options:
      raise NotImplementedError('options not supported with the gym wrapper.')
    del options
    observation = self._csuite_env.start(seed)
    state = self._csuite_env.get_state()
    self._np_random = state.rng if hasattr(
        state, 'rng') else np.random.default_rng(seed)
    if gym.__version__ == '0.19.0':
      return observation
    else:
      return observation, {}

  def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:

    if mode == 'rgb_array':
      return self._csuite_env.render()

    if mode == 'human':
      if self.viewer is None:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=g-import-not-at-top
        from gym.envs.classic_control import rendering
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(self._csuite_env.render())
      return self.viewer.isopen

  @property
  def action_space(self) -> spaces.Discrete:
    action_spec = self._csuite_env.action_spec()
    if isinstance(action_spec, specs.DiscreteArray):
      action_spec = typing.cast(specs.DiscreteArray, action_spec)
      return spaces.Discrete(action_spec.num_values)
    else:
      raise NotImplementedError(
          'The gym wrapper only supports environments with discrete action '
          'spaces. Please raise an issue if you want to work with a '
          'a non-discrete action space.')

  @property
  def observation_space(self) -> spaces.Box:
    obs_spec = self._csuite_env.observation_spec()
    if isinstance(obs_spec, specs.BoundedArray):
      return spaces.Box(
          low=float(obs_spec.minimum),
          high=float(obs_spec.maximum),
          shape=obs_spec.shape,
          dtype=obs_spec.dtype)
    return spaces.Box(
        low=-float('inf'),
        high=float('inf'),
        shape=obs_spec.shape,
        dtype=obs_spec.dtype)

  @property
  def reward_range(self) -> Tuple[float, float]:
    # CSuite does not return reward range.
    return -float('inf'), float('inf')

  def __getattr__(self, attr):
    """Delegate attribute access to underlying environment."""
    return getattr(self._csuite_env, attr)
