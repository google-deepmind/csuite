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

"""Abstract base class for csuite environments."""

import abc
from typing import Any, Tuple

from dm_env import specs


# TODO(b/243715530): The base environment should implementing this check.
STEP_WITHOUT_START_ERR = ("Environment state has not been initialized. `start`"
                          " must be called before calling `step`.")


class Environment(abc.ABC):
  """Base class for continuing environments.

  Observations and valid actions are described by the `specs` module in dm_env.
  Environment implementations should return specs as specific as possible.

  Each environment will specify its own environment State, Configuration, and
  internal random number generator.
  """

  @abc.abstractmethod
  def start(self) -> Any:
    """Starts (or restarts) the environment by setting the initial state.

    Returns an initial observation.
    """

  @abc.abstractmethod
  def step(self, action: Any) -> Tuple[Any, Any]:
    """Takes a step in the environment, returning an observation and reward."""

  @abc.abstractmethod
  def observation_spec(self) -> specs.Array:
    """Describes the observation space of the environment.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.
    """

  @abc.abstractmethod
  def action_spec(self) -> specs.Array:
    """Describes the valid action space of the environment.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.
    """

  @abc.abstractmethod
  def get_state(self) -> Any:
    """Returns the environment state."""

  @abc.abstractmethod
  def set_state(self, state: Any):
    """Sets the environment state."""

  @abc.abstractmethod
  def render(self) -> Any:
    """Returns an object (e.g. a numpy array) to facilitate visualization."""
