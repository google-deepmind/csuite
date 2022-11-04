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

"""Implementation of the tabular Taxi environment.

Environment description and details can be found in the `Taxi`
environment class.

The 5x5 gridworld is depicted below, where we use xy-coordinates to describe the
position of the squares; the coordinate (0, 0) represents the top left corner.

|||||||||||
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
|||||||||||
"""

import copy
import dataclasses
import enum
import itertools
from typing import Optional

from csuite.environments import base
from dm_env import specs

import numpy as np
from PIL import Image
from PIL import ImageDraw

# Default environment variables.
_NUM_ROWS = 5
_NUM_COLUMNS = 5
# Passenger positions: one of the four pickup locations, or in the taxi.
_NUM_POSITIONS = 5
_NUM_DEST = 4
_NUM_STATES = _NUM_ROWS * _NUM_COLUMNS * _NUM_POSITIONS * _NUM_DEST
_NUM_ACTIONS = 6

# Error messages.
_INVALID_ACTION = "Invalid action: expected value in [0,5] but received {}."
_INVALID_TAXI_LOC = "Invalid state: expected taxi coordinates in range [0,4]."
_INVALID_PASS_LOC = ("Invalid state: expected passenger location as an integer"
                     "in [0,4].")
_INVALID_DEST = ("Invalid state: expected destination location as an integer"
                 "in [0,3].")

# Dictionary mapping the four colored squares to their xy-coordinate
# on the 5x5 grid, with keys 0 (Red), 1 (Green), 2 (Yellow), 3 (Blue).
_COLOR_POSITIONS = {
    0: (0, 0),
    1: (4, 0),
    2: (0, 4),
    3: (3, 4),
}

# List of (x, y) pairs where transitioning
# between each pair's shared edge is forbidden.
_BLOCKED_TUPLE = (
    ((1, 0), (2, 0)),
    ((1, 1), (2, 1)),
    ((0, 3), (1, 3)),
    ((0, 4), (1, 4)),
    ((2, 3), (3, 3)),
    ((2, 4), (3, 4)),
)
# Variables for pixel visualization of the environment.
_PIXELS_PER_SQ = 50  # size of each grid square.
_RED_HEX = "#ff9999"
_GREEN_HEX = "#9cff9c"
_BLUE_HEX = "#99e2ff"
_YELLOW_HEX = "#fff899"
_PASS_LOC_HEX = "#d400ff"
_DEST_HEX = "#008c21"
_EMPTY_TAXI_HEX = "#8f8f8f"

# Other derived constants used for pixel visualization.
_HEIGHT = _PIXELS_PER_SQ * (_NUM_ROWS + 2)
_WIDTH = _PIXELS_PER_SQ * (_NUM_COLUMNS + 2)
_BORDER = [  # Bounding box for perimeter of the taxi grid.
    (_PIXELS_PER_SQ, _PIXELS_PER_SQ),
    (_PIXELS_PER_SQ * (_NUM_COLUMNS + 1), _PIXELS_PER_SQ * (_NUM_ROWS + 1))
]
_OFFSET = _PIXELS_PER_SQ // 5  # To make the taxi bounding box slightly smaller.
_LINE_WIDTH_THIN = _PIXELS_PER_SQ // 50
_LINE_WIDTH_THICK = _PIXELS_PER_SQ // 10

# Dictionary mapping the four colored squares to their rectangle bounding boxes
# used for visualization, with keys 0 (Red), 1 (Green), 2 (Yellow), 3 (Blue).
_BOUNDING_BOXES = {
    idx: [(_PIXELS_PER_SQ * (x + 1), _PIXELS_PER_SQ * (y + 1)),
          (_PIXELS_PER_SQ * (x + 2), _PIXELS_PER_SQ * (y + 2))]
    for idx, (x, y) in _COLOR_POSITIONS.items()
}


class Action(enum.IntEnum):
  """Actions for the Taxi environment.

  There are six actions:
  0: move North.
  1: move West.
  2: move South.
  3: move East.
  4: pickup the passenger.
  5: dropoff the passenger.
  """
  NORTH, WEST, SOUTH, EAST, PICKUP, DROPOFF = list(range(_NUM_ACTIONS))

  @property
  def dx(self):
    """Maps EAST to 1, WEST to -1, and other actions to 0."""
    if self.name == "EAST":
      return 1
    elif self.name == "WEST":
      return -1
    else:
      return 0

  @property
  def dy(self):
    """Maps NORTH to -1, SOUTH to 1, and other actions to 0."""
    if self.name == "NORTH":
      return -1
    elif self.name == "SOUTH":
      return 1
    else:
      return 0


@dataclasses.dataclass
class State:
  """State of a continuing Taxi environment.

  The coordinate system excludes border of the map and provides the location of
  the taxi. The coordinate (0, 0) corresponds to the top left corner,
  i.e. the Red pickup location, and the coordinate (4, 4) corresponds to the
  bottom right corner.

  The passenger location is an integer in [0, 4] corresponding to the four
  colored squares and the fifth possible position being in the taxi. The
  destination is similarly numbered, but only includes the four colored squares:
  0 - Red square.
  1 - Green square.
  2 - Yellow square.
  3 - Blue square.
  4 - In taxi.
  """
  taxi_x: int
  taxi_y: int
  passenger_loc: int
  destination: int
  rng: np.random.Generator


class Taxi(base.Environment):
  """A continuing Taxi environment.

  This environment originates from the paper "Hierarchical Reinforcement
  Learning with the MAXQ Value Function Decomposition" by Tom Dietterich.

  In a square grid world with four colored squares (R(ed), G(reen), Y(ellow),
  B(lue)), the agent must drive a taxi to various passengers' locations and
  drop them off at the passenger's desired location at one of the four squares.
  The agent receives positive reward for each passenger successfully picked up
  and dropped off, and receives negative reward for doing an inappropriate
  action (eg. dropping off the passenger at the incorrect location, attempting
  to pick up a passenger on an empty square, etc.).

  There are six possible actions, corresponding to four navigation actions
  (move North, South, East, and West), a pickup action, and a dropoff action.

  The observation space is a single state index, which encodes the possible
  states accounting for the taxi position, location of the passenger, and four
  desired destination locations.
  """

  def __init__(self, seed=None):
    """Initialize Taxi environment.

    Args:
      seed: Seed for the internal random number generator.
    """
    self._seed = seed
    self._state = None

    # Populate lookup table for observations.
    self.lookup_table = {}
    for idx, state in enumerate(
        itertools.product(
            range(_NUM_ROWS), range(_NUM_COLUMNS), range(_NUM_POSITIONS),
            range(_NUM_DEST))):
      self.lookup_table[state] = idx

  def start(self, seed: Optional[int] = None):
    """Initializes the environment and returns an initial observation."""
    rng = np.random.default_rng(self._seed if seed is None else seed)
    self._state = State(
        taxi_x=rng.integers(_NUM_COLUMNS),
        taxi_y=rng.integers(_NUM_ROWS),
        passenger_loc=rng.integers(_NUM_POSITIONS - 1),
        destination=rng.integers(_NUM_DEST),
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
      action: An integer in [0,5] indicating whether the taxi moves, picks up
        the passenger, or drops off the passenger.

    Returns:
      A tuple of type (int, float) giving the next observation and the reward.

    Raises:
      RuntimeError: If state has not yet been initialized by `start`.
      ValueError: If input action has an invalid value.
    """
    # Check if state has been initialized.
    if not self.started:
      raise RuntimeError(base.STEP_WITHOUT_START_ERR)

    # Check if input action is valid.
    if action not in [a.value for a in Action]:
      raise ValueError(_INVALID_ACTION.format(action))

    reward = 0
    # Move taxi according to the action.
    self._state.taxi_y = np.clip(self._state.taxi_y + Action(action).dy, 0,
                                 _NUM_ROWS - 1)
    # If moving East or West, check that the taxi does not hit a barrier.
    if action in [Action.EAST, Action.WEST]:
      move = ((self._state.taxi_x, self._state.taxi_y),
              (self._state.taxi_x + Action(action).dx, self._state.taxi_y))
      if action == Action.WEST:  # Need to reverse the tuple.
        move = move[::-1]
      if move not in _BLOCKED_TUPLE:
        self._state.taxi_x = np.clip(self._state.taxi_x + Action(action).dx, 0,
                                     _NUM_COLUMNS - 1)

    # If action is pickup, check if passenger location matches current location.
    if action == Action.PICKUP:
      if self._state.passenger_loc == 4:  # Passenger was already picked up.
        reward = -10
      else:
        passenger_coordinates = _COLOR_POSITIONS[self._state.passenger_loc]
        # Check if passenger and taxi are at the same location.
        if passenger_coordinates != (self._state.taxi_x, self._state.taxi_y):
          reward = -10
        else:
          # Passenger has been successfully picked up.
          self._state.passenger_loc = 4

    # If action is dropoff, check if passenger is present in taxi and
    # desired destination matches current location.
    if action == Action.DROPOFF:
      dest_coordinates = _COLOR_POSITIONS[self._state.destination]
      if (self._state.passenger_loc != 4 or dest_coordinates !=
          (self._state.taxi_x, self._state.taxi_y)):
        reward = -10
      else:
        reward = 20
        # Add new passenger.
        self._state.passenger_loc = self._state.rng.integers(_NUM_POSITIONS - 1)
        self._state.destination = self._state.rng.integers(_NUM_DEST)

    return self._get_observation(), reward

  def _get_observation(self):
    """Returns a observation index uniquely identifying the current state."""
    state_tuple = (self._state.taxi_x, self._state.taxi_y,
                   self._state.passenger_loc, self._state.destination)
    return self.lookup_table[state_tuple]

  def observation_spec(self):
    """Describes the observation specs of the environment."""
    return specs.DiscreteArray(_NUM_STATES, dtype=int, name="observation")

  def action_spec(self):
    """Describes the action specs of the environment."""
    return specs.DiscreteArray(_NUM_ACTIONS, dtype=int, name="action")

  def get_state(self):
    """Returns a copy of the current environment state."""
    return copy.deepcopy(self._state) if self._state is not None else None

  def set_state(self, state):
    """Sets environment state to state provided.

    Args:
      state: A State object which overrides the current state.
    """
    # Check that input state values are valid.
    if not (0 <= state.taxi_x < _NUM_COLUMNS and 0 <= state.taxi_y < _NUM_ROWS):
      raise ValueError(_INVALID_TAXI_LOC)
    elif not 0 <= state.passenger_loc < _NUM_POSITIONS:
      raise ValueError(_INVALID_PASS_LOC)
    elif not 0 <= state.destination < _NUM_DEST:
      raise ValueError(_INVALID_DEST)

    self._state = copy.deepcopy(state)

  def render(self) -> np.ndarray:
    """Creates an image of the current environment state.

    The underlying grid with the four colored squares are drawn. The taxi is
    drawn as a circle which is _PASS_LOC_HEX (purple) when the passenger is
    present, and _EMPTY_TAXI_HEX (grey) otherwise. The passenger location before
    being picked up is outlined with the color _PASS_LOC_HEX (purple), and the
    destination location is similarly outlined with the color _DEST_HEX
    (dark green).

    In the case where the passenger location and self._state.destination are
    identical, only the self._state.destination outline is visible.

    Returns:
      A NumPy array giving an image of the environment state.
    """
    image = Image.new("RGB", (_HEIGHT, _WIDTH), "white")
    dct = ImageDraw.Draw(image)

    # First place four colored destination squares so grid lines appear on top.
    # Red, green, yellow, and blue squares.
    dct.rectangle(_BOUNDING_BOXES[0], fill=_RED_HEX)
    dct.rectangle(_BOUNDING_BOXES[1], fill=_GREEN_HEX)
    dct.rectangle(_BOUNDING_BOXES[2], fill=_YELLOW_HEX)
    dct.rectangle(_BOUNDING_BOXES[3], fill=_BLUE_HEX)

    # Draw basic grid.
    for row in range(1, _NUM_ROWS + 2):  # horizontal grid lines.
      line_coordinates = [(_PIXELS_PER_SQ, _PIXELS_PER_SQ * row),
                          (_PIXELS_PER_SQ * (_NUM_ROWS + 1),
                           _PIXELS_PER_SQ * row)]
      dct.line(line_coordinates, fill="black", width=_LINE_WIDTH_THIN)
    for col in range(1, _NUM_COLUMNS + 2):  # vertical grid lines.
      line_coordinates = [(_PIXELS_PER_SQ * col, _PIXELS_PER_SQ),
                          (_PIXELS_PER_SQ * col,
                           _PIXELS_PER_SQ * (_NUM_ROWS + 1))]
      dct.line(line_coordinates, fill="black", width=_LINE_WIDTH_THIN)

    # Draw barriers.
    dct.rectangle(
        _BORDER,  # Grid perimeter.
        outline="black",
        width=_LINE_WIDTH_THICK)

    def get_barrier_coordinates(x, y):
      """Returns bounding box for barrier (length two down from input)."""
      return [(x, y), (x, y + 2 * _PIXELS_PER_SQ)]

    # Top barrier, bottom left barrier, bottom right barrier.
    dct.line(
        get_barrier_coordinates(3 * _PIXELS_PER_SQ, _PIXELS_PER_SQ),
        fill="black",
        width=_LINE_WIDTH_THICK)
    dct.line(
        get_barrier_coordinates(2 * _PIXELS_PER_SQ, 4 * _PIXELS_PER_SQ),
        fill="black",
        width=_LINE_WIDTH_THICK)
    dct.line(
        get_barrier_coordinates(4 * _PIXELS_PER_SQ, 4 * _PIXELS_PER_SQ),
        fill="black",
        width=_LINE_WIDTH_THICK)

    # Draw passenger location.
    if self._state.passenger_loc in range(4):
      taxi_color = _EMPTY_TAXI_HEX
      dct.rectangle(
          _BOUNDING_BOXES[self._state.passenger_loc],
          outline=_PASS_LOC_HEX,
          width=_LINE_WIDTH_THICK)
    else:
      taxi_color = _PASS_LOC_HEX

    # Draw taxi.
    def get_circle_coordinates(x, y):
      return [((x + 1) * _PIXELS_PER_SQ + _OFFSET,
               (y + 1) * _PIXELS_PER_SQ + _OFFSET),
              ((x + 2) * _PIXELS_PER_SQ - _OFFSET,
               (y + 2) * _PIXELS_PER_SQ - _OFFSET)]

    dct.ellipse(
        get_circle_coordinates(self._state.taxi_x, self._state.taxi_y),
        fill=taxi_color)

    # Draw self._state.destination location.
    dct.rectangle(
        _BOUNDING_BOXES[self._state.destination],
        outline=_DEST_HEX,
        width=_LINE_WIDTH_THICK)
    return np.asarray(image, dtype=np.uint8)
