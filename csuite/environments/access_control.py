"""Implementation of the tabular Access-Control environment.

Environment description and details can be found in the `AccessControl`
environment class.
"""

import copy
import dataclasses
import enum
import itertools

from csuite.environments import base
from dm_env import specs

import numpy as np

# Default environment variables from Sutton&Barto Example 10.2.
_NUM_SERVERS = 10
_FREE_PROBABILITY = 0.06
_PRIORITIES = (1, 2, 4, 8)

# Error messages.
_STEP_WITHOUT_START = ("Environment state has not been initialized."
                       "`start` must be called before calling `step`.")
_INVALID_ACTION = "Invalid action: expected 0 or 1 but received {action}."
_INVALID_BUSY_SERVERS = ("Invalid state: num_busy_servers not in expected"
                         "range [0, {}).")
_INVALID_PRIORITY = ("Invalid state: incoming_priority not in expected"
                     "range {}.")


class Action(enum.IntEnum):
  REJECT = 0
  ACCEPT = 1


@dataclasses.dataclass
class Params:
  """Parameters of an Access-Control instance.

  Attributes:
    num_servers: A positive integer, denoting the total number of available
      servers.
    free_probability: A positive float, denoting the probability a busy server
     becomes free at each timestep.
    priorities: A list of floats, giving the possible priorities of incoming
      customers.
  """
  num_servers: int
  free_probability: float
  priorities: list[float]


@dataclasses.dataclass
class State:
  """State of an Access-Control continuing environment.

  Let N be the number of servers.

  Attributes:
    num_busy_servers: An integer in the range [0, N] representing the number of
      busy servers.
    incoming_priority: An integer giving the priority of the incoming customer.
  """
  num_busy_servers: int
  incoming_priority: int


class AccessControl(base.Environment):
  """An Access-Control continuing environment.

  Given access to a set of servers and an infinite queue of customers
  with different priorities, the agent must decide whether to accept
  or reject the next customer in line based on their priority and the
  number of free servers.

  There are two actions: accept or decline the incoming customer. Note that
  if no servers are available, the customer is declined regardless of the
  action selected.

  The observation is a single state index, enumerating the possible states
  (num_busy_servers, incoming_priority).

  The default environment follows that described in Sutton and Barto's book
  (Example 10.2 in the second edition).
  """

  def __init__(self,
               num_servers=_NUM_SERVERS,
               free_probability=_FREE_PROBABILITY,
               priorities=_PRIORITIES,
               seed=None):
    """Initialize Access-Control environment.

    Args:
      num_servers: A positive integer, denoting the total number of available
        servers.
      free_probability: A positive float, denoting the probability a busy server
        becomes free at each timestep.
      priorities: A list of floats, giving the possible priorities of incoming
        customers.
      seed: Seed for the internal random number generator.
    """
    self._rng = np.random.RandomState(seed)
    self._params = Params(
        num_servers=num_servers,
        free_probability=free_probability,
        priorities=priorities
    )
    self.num_states = ((self._params.num_servers + 1)
                       * len(self._params.priorities))

    # Populate lookup table for observations.
    self.lookup_table = {}
    for idx, state in enumerate(itertools.product(
        range(self._params.num_servers + 1), self._params.priorities)):
      self.lookup_table[state] = idx

    self._state = None

  def start(self):
    """Initializes the environment and returns an initial observation."""
    self._state = State(
        num_busy_servers=0,
        incoming_priority=self._rng.choice(self._params.priorities)
    )
    return self._get_observation()

  def step(self, action):
    """Updates the environment state and returns an observation and reward.

    Args:
      action: An integer equalling 0 or 1 to reject or accept the customer.

    Returns:
      A tuple of type (int, float) giving the next observation and the reward.

    Raises:
      RuntimeError: If state has not yet been initialized by `start`.
      ValueError: If input action has an invalid value.
    """
    # Check if state has been initialized.
    if self._state is None:
      raise RuntimeError(_STEP_WITHOUT_START)

    # Check if input action is valid.
    if action not in [Action.REJECT, Action.ACCEPT]:
      raise ValueError(_INVALID_ACTION.format(action=action))

    reward = 0
    # If customer is accepted, ensure there are enough free servers.
    if (action == Action.ACCEPT and
        self._state.num_busy_servers < self._params.num_servers):
      reward = self._state.incoming_priority
      self._state.num_busy_servers += 1

    new_priority = self._rng.choice(self._params.priorities)

    # Update internal state by freeing busy servers with a given probability.
    num_busy_servers = self._state.num_busy_servers
    num_new_free_servers = self._rng.binomial(num_busy_servers,
                                              p=self._params.free_probability)
    self._state.num_busy_servers = num_busy_servers - num_new_free_servers
    self._state.incoming_priority = new_priority

    return self._get_observation(), reward

  def _get_observation(self):
    """Converts internal state to an index uniquely identifying the state.

    Returns:
      An integer denoting the current state's index according to the
      enumeration of the state space stored by the environment's lookup table.
    """
    state_key = (self._state.num_busy_servers, self._state.incoming_priority)
    return self.lookup_table[state_key]

  def observation_spec(self):
    """Describes the observation specs of the environment."""
    return specs.DiscreteArray(self.num_states, dtype=int,
                               name="observation")

  def action_spec(self):
    """Describes the action specs of the environment."""
    return specs.DiscreteArray(2, dtype=int, name="action")

  def get_state(self):
    """Returns a copy of the current environment state."""
    return copy.deepcopy(self._state) if self._state is not None else None

  def set_state(self, state):
    """Sets environment state to state provided.

    Args:
      state: A State object which overrides the current state.
    """
    # Check that input state values are valid.
    if state.num_busy_servers not in range(self._params.num_servers + 1):
      raise ValueError(_INVALID_BUSY_SERVERS.format(self._params.num_servers))
    elif state.incoming_priority not in self._params.priorities:
      raise ValueError(_INVALID_PRIORITY.format(self._params.priorities))

    self._state = copy.deepcopy(state)

  def get_config(self):
    """Returns a copy of the environment configuration."""
    return copy.deepcopy(self._params)

  def render(self):
    return np.array(
        [self._state.num_busy_servers, self._state.incoming_priority])
