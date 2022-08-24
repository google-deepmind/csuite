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

"""Continuing catch environment with non-stationary permutations of the observation features.

Environment description can be found in the `DancingCatch` environment class.
"""
import copy
import dataclasses
import enum

from csuite.environments import base
from dm_env import specs

import numpy as np

# Error messages.
_STEP_WITHOUT_START = ("Environment state has not been initialized. `start`"
                       " must be called before calling `step`.")
_INVALID_ACTION = "Invalid action: expected 0, 1, or 2 but received {action}."
_INVALID_PADDLE_POS = ("Invalid state: paddle should be positioned at the"
                       " bottom of the board.")
_INVALID_BALLS_RANGE = (
    "Invalid state: positions of balls and paddle not in expected"
    " row range [0, {rows}) and column range [0, {columns})."
)

# Default environment variables.
_ROWS = 10
_COLUMNS = 5
_SPAWN_PROBABILITY = 0.1
_SWAP_EVERY = 10000


class Action(enum.IntEnum):
  LEFT = 0
  STAY = 1
  RIGHT = 2

  @property
  def dx(self):
    """Maps LEFT to -1, STAY to 0 and RIGHT to 1."""
    return self.value - 1


@dataclasses.dataclass
class Params:
  """Parameters of a continuing Catch instance.

  Attributes:
    rows: Integer number of rows.
    columns: Integer number of columns.
    observation_dim: Integer dimension of the observation features.
    spawn_probability: Probability of a new ball spawning.
    swap_every: Integer giving the interval at which a swap occurs.
  """
  rows: int
  columns: int
  observation_dim: int
  spawn_probability: float
  swap_every: int


@dataclasses.dataclass
class State:
  """State of a continuing Catch instance.

  Attributes:
    paddle_x: An integer denoting the x-coordinate of the paddle.
    paddle_y: An integer denoting the y-coordinate of the paddle
    balls: A list of (x, y) coordinates representing the present balls.
    shuffle_idx: Indices for performing the observation shuffle as a result of
      the random swaps.
    time_since_swap: An integer denoting how many timesteps have elapsed
      since the last swap.
    rng: Internal NumPy pseudo-random number generator, included here for
      reproducibility purposes.
  """
  paddle_x: int
  paddle_y: int
  balls: list[tuple[int, int]]
  shuffle_idx: np.ndarray
  time_since_swap: int
  rng: np.random.RandomState


class DancingCatch(base.Environment):
  """A continuing Catch environment with random swaps.

  This environment is the same as the continuing Catch environment, but
  at each timestep, there is a low probability that two entries in the
  observation are swapped.

  The observations are flattened, with entry one if the corresponding unshuffled
  position contains the paddle or a ball, and zero otherwise.
  """

  def __init__(self,
               rows=_ROWS,
               columns=_COLUMNS,
               spawn_probability=_SPAWN_PROBABILITY,
               seed=None,
               swap_every=_SWAP_EVERY):
    """Initializes a continuing Catch environment.

    Args:
      rows: A positive integer denoting the number of rows.
      columns: A positive integer denoting the number of columns.
      spawn_probability: Float giving the probability of a new ball appearing.
      seed: Seed for the internal random number generator.
      swap_every: A positive integer denoting the interval at which a swap in
        the observation occurs.
    """
    self._seed = seed
    self._params = Params(rows=rows,
                          columns=columns,
                          observation_dim=rows * columns,
                          spawn_probability=spawn_probability,
                          swap_every=swap_every)
    self._state = None

  def start(self):
    """Initializes the environment and returns an initial observation."""

    # The initial state has one ball appearing in a random column at the top,
    # and the paddle centered at the bottom.

    rng = np.random.RandomState(self._seed)
    self._state = State(
        paddle_x=self._params.columns // 2,
        paddle_y=self._params.rows - 1,
        balls=[(rng.randint(self._params.columns), 0)],
        shuffle_idx=np.arange(self._params.observation_dim),
        time_since_swap=0,
        rng=rng,
    )
    return self._get_observation()

  @property
  def started(self):
    """True if the environment has been started, False otherwise."""
    # An unspecified state implies that the environment needs to be started.
    return self._state is not None

  def step(self, action):
    """Updates the environment state and returns an observation and reward.

    Args:
      action: An integer equalling 0, 1, or 2 indicating whether to move the
        paddle left, stay, or move the paddle right respectively.

    Returns:
      A tuple of type (int, float) giving the next observation and the reward.

    Raises:
      RuntimeError: If state has not yet been initialized by `start`.
    """
    # Check if state has been initialized.
    if not self.started:
      raise RuntimeError(_STEP_WITHOUT_START)

    # Check if input action is valid.
    if action not in [Action.LEFT, Action.STAY, Action.RIGHT]:
      raise ValueError(_INVALID_ACTION.format(action=action))

    # Move the paddle.
    self._state.paddle_x = np.clip(self._state.paddle_x + Action(action).dx,
                                   0, self._params.columns - 1)

    # Move all balls down by one unit.
    self._state.balls = [(x, y + 1) for x, y in self._state.balls]

    # Since at most one ball is added at each timestep, at most one ball
    # can be at the bottom of the board, and must be the 'oldest' ball.
    reward = 0.
    if self._state.balls and self._state.balls[0][1] == self._state.paddle_y:
      reward = 1. if self._state.balls[0][0] == self._state.paddle_x else -1.
      # Remove ball from list.
      self._state.balls = self._state.balls[1:]

    # Add new ball with given probability.
    if self._state.rng.random() < self._params.spawn_probability:
      self._state.balls.append(
          (self._state.rng.randint(self._params.columns), 0)
      )

    # Update time since last swap.
    self._state.time_since_swap += 1

    # Update the observation permutation indices by swapping two indices,
    # at the given interval.
    if self._state.time_since_swap % self._params.swap_every == 0:
      idx_1, idx_2 = self._state.rng.randint(self._params.observation_dim,
                                             size=2).T
      self._state.shuffle_idx[[idx_1, idx_2]] = (
          self._state.shuffle_idx[[idx_2, idx_1]]
      )
      self._state.time_since_swap = 0

    return self._get_observation(), reward

  def _get_observation(self) -> np.ndarray:
    """Converts internal environment state to an array observation.

    Returns:
      A binary array of size (rows, columns) with entry 1 if it contains either
      a ball or a paddle, and entry 0 if the cell is empty.
    """
    board = np.zeros((_ROWS, _COLUMNS), dtype=int)
    board.fill(0)
    board[self._state.paddle_y, self._state.paddle_x] = 1
    for x, y in self._state.balls:
      board[y, x] = 1
    board = board.flatten()
    return board[self._state.shuffle_idx]

  def observation_spec(self):
    """Describes the observation specs of the environment."""
    return specs.BoundedArray(shape=(self._params.observation_dim,),
                              dtype=int, minimum=0, maximum=1, name="board")

  def action_spec(self):
    """Describes the action specs of the environment."""
    return specs.DiscreteArray(num_values=3, dtype=int, name="action")

  def get_state(self):
    """Returns a copy of the current environment state."""
    return copy.deepcopy(self._state) if self._state is not None else None

  def set_state(self, state):
    """Sets environment state to state provided.

    Args:
      state: A State object which overrides the current state.
    """
    # Check that input state values are valid.
    if  not (0 <= state.paddle_x < self._params.columns
             and state.paddle_y == self._params.rows - 1):
      raise ValueError(_INVALID_PADDLE_POS)

    for x, y in state.balls:
      if not (0 <= x < self._params.columns and 0 <= y < self._params.rows):
        raise ValueError(_INVALID_BALLS_RANGE.format(
            rows=self._params.rows, columns=self._params.columns))

    self._state = copy.deepcopy(state)

  def get_config(self):
    """Returns a copy of the environment configuration."""
    return copy.deepcopy(self._params)

  def render(self):
    return self._get_observation()
