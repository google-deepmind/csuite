"""Abstract base class for csuite environments."""

import abc
from typing import Any

from dm_env import specs

# TODO(rosiezhao): Make observation and action types more specific.


class Environment(abc.ABC):
  """Base class for continuing environments.

  Observations and valid actions are described by the `specs` module in dm_env.
  Each environment will specify its own environment State, Configuration, and
  internal random number generator.
  """

  @abc.abstractmethod
  def start(self) -> Any:
    """Starts the environment by setting the initial state.

    Returns an initial observation.
    """

  @abc.abstractmethod
  def step(self, action: Any) -> Any:
    """Takes a step in the environment, returning an observation and reward."""

  @abc.abstractmethod
  def observation_spec(self) -> specs.Array:
    """Describes the observation space of the environment.

    Returns an `Array` object from dm_env.specs.
    """

  @abc.abstractmethod
  def action_spec(self) -> specs.Array:
    """Describes the valid action space of the environment.

    Returns an `Array` object from dm_env.specs.
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
