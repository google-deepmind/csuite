"""Implementation of the 1D continuous-action version 
of the PuckWorld environment with movement just along the x-axis.
"""

import math
import numpy as np
import copy
import dataclasses
import enum
from typing import Optional

from csuite.environments import base
from dm_env import specs

from PIL import Image
from PIL import ImageDraw

_NUM_ACTIONS = 4
_WIDTH = 5
_HEIGHT = 5
_ACCELERATION = 0.1
_FRICTION = 0.5
_PUCK_RADIUS = 0.5
_GOAL_RADIUS = 0.25
_GOAL_UPDATE_INTERVAL = 300
_SIMULATION_STEP_SIZE = 0.05
_ACT_STEP_PERIOD = 4

# Error messages.
_INVALID_ACTION = "Invalid action: expected value in [0,3] but received {}."
_INVALID_POSITION = "Invalid position: expected value in [0,{}]"
_INVALID_VELOCITY = "Invalid velocity: expected value in [{},{}]"

# Variables for pixel visualization of the environment.
_PIXELS_PER_SQ = 50  # size of each grid square.
_RED_HEX = "#ff9999"
_GREEN_HEX = "#9cff9c"
_IMAGE_WIDTH = _PIXELS_PER_SQ * _WIDTH
_IMAGE_HEIGHT = _PIXELS_PER_SQ * _HEIGHT


@dataclasses.dataclass
class State:
  """State of the PuckWorld environment.

  The coordinate system provides the location of the puck and the goal.
  (0, 0) corresponds to the top left corner.
  """
  puck_pos_x: float
  puck_pos_y: float
  puck_vel_x: float
  puck_vel_y: float
  goal_pos_x: float
  goal_pos_y: float
  rng: np.random.Generator


@dataclasses.dataclass
class Params:
  goal_update_interval: float
  friction: float
  puck_radius: float
  simulation_step_size: float
  act_step_period: int
  max_velocity: float


class PuckWorldContinuous1D(base.Environment):

  def __init__(self,
               puck_radius=_PUCK_RADIUS,
               friction=_FRICTION,
               goal_update_interval=_GOAL_UPDATE_INTERVAL,
               simulation_step_size=_SIMULATION_STEP_SIZE,
               act_step_period=_ACT_STEP_PERIOD,
               seed=None):
    """Initialize PuckWorld environment.

        Args:
          seed: Seed for the internal random number generator.
    """
    self._seed = seed
    self._state = None
    self._counter = None
    self._params = Params(
      puck_radius=puck_radius,
      friction=friction,
      goal_update_interval=goal_update_interval,
      simulation_step_size=simulation_step_size,
      act_step_period=act_step_period,
      max_velocity=(_ACCELERATION / friction)
    )

  def start(self, seed: Optional[int] = None):
    """Initializes the environment and returns an initial observation."""
    rng = np.random.default_rng(self._seed if seed is None else seed)
    self._counter = 0
    self._state = State(
      puck_pos_x = 0.5 * _WIDTH,
      puck_pos_y = 0.5 * _HEIGHT,
      puck_vel_x = 0,
      puck_vel_y = 0,
      goal_pos_x = rng.random() * _WIDTH,
      goal_pos_y = 0.5 * _HEIGHT,
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
      action: a 2d array of elements in [-1,1] indicating the action in x and y directions

    Returns:
      A tuple of type (np.ndarray, float) giving the next observation
      and the reward.

    Raises:
      RuntimeError: If state has not yet been initialized by `start`.
      ValueError: If input action has an invalid value.
    """
    # Check if state has been initialized.
    if not self.started:
      raise RuntimeError(base.STEP_WITHOUT_START_ERR)

    # Check if input action is valid.
    assert action.all() >= -1 and action.all() <= 1, f"Invalid action: expected values in [-1,1] but received {action}."

    self._counter += 1

    for _ in range(self._params.act_step_period):
        # Update puck's velocity according to the action.
        delta_vel_x = ((action[0] * _ACCELERATION -
                        self._params.friction * self._state.puck_vel_x) *
                       self._params.simulation_step_size)
        delta_vel_y = ((action[1] * _ACCELERATION -
                        self._params.friction * self._state.puck_vel_y) *
                       self._params.simulation_step_size)
        self._state.puck_vel_x += delta_vel_x
        self._state.puck_vel_y += delta_vel_y

        # Update puck's position.
        self._state.puck_pos_x += (self._state.puck_vel_x *
                                   self._params.simulation_step_size)
        self._state.puck_pos_y += (self._state.puck_vel_y *
                                   self._params.simulation_step_size)

    # Check for boundary collisions.
    if self._state.puck_pos_x < self._params.puck_radius:
      self._state.puck_vel_x *= -0.5
      self._state.puck_pos_x = self._params.puck_radius
    if self._state.puck_pos_y < self._params.puck_radius:
      self._state.puck_vel_y *= -0.5
      self._state.puck_pos_y = self._params.puck_radius
    if self._state.puck_pos_x > _WIDTH - self._params.puck_radius:
      self._state.puck_vel_x *= -0.5
      self._state.puck_pos_x = _WIDTH - self._params.puck_radius
    if self._state.puck_pos_y > _HEIGHT - self._params.puck_radius:
      self._state.puck_vel_y *= -0.5
      self._state.puck_pos_y = _HEIGHT - self._params.puck_radius

    # Update goal after a fixed interval.
    if self._counter % self._params.goal_update_interval == 0:
      self._update_goal()

    # Compute reward.
    distance_from_goal = self._compute_distance_from_goal()
    reward = -distance_from_goal

    return self._get_observation(), reward

  def _get_observation(self):
    return np.array((self._state.puck_pos_x,
                     self._state.puck_vel_x,
                     self._state.goal_pos_x))

  def _update_goal(self):
    self._state.goal_pos_x = self._state.rng.random() * _WIDTH
    self._state.goal_pos_y = 0.5 * _HEIGHT

  def _compute_distance_from_goal(self):
    dx = self._state.puck_pos_x - self._state.goal_pos_x
    dy = self._state.puck_pos_y - self._state.goal_pos_y
    return math.sqrt(dx**2 + dy**2)

  def observation_spec(self):
    """Describes the observation specs of the environment."""
    return specs.BoundedArray((3,),         # don't even need to know the velocity
                              dtype=np.float32,
                              minimum=[0, -self._params.max_velocity, 0],
                              maximum=[_WIDTH, self._params.max_velocity, _WIDTH])

  def action_spec(self):
    """Describes the action specs of the environment."""
    return specs.BoundedArray((1,), dtype=np.float32, minimum=-1, maximum=1, name="action")

  def get_counter(self):
    """Returns the current value of the counter."""
    return self._counter

  def get_state(self):
    """Returns a copy of the current environment state."""
    return copy.deepcopy(self._state) if self._state is not None else None

  def set_state(self, state):
    """Sets environment state to state provided.

    Args:
      state: A State object which overrides the current state.

    Returns:
      A NumPy array for the observation.
    """
    # Check that input state values are valid.
    if not 0 <= state.puck_pos_x <= _WIDTH:
      raise ValueError(_INVALID_POSITION.format(_WIDTH))
    if not 0 <= state.puck_pos_y <= _HEIGHT:
      raise ValueError(_INVALID_POSITION.format(_HEIGHT))
    if not 0 <= state.goal_pos_x <= _WIDTH:
      raise ValueError(_INVALID_POSITION.format(_WIDTH))
    if not 0 <= state.goal_pos_y <= _HEIGHT:
      raise ValueError(_INVALID_POSITION.format(_HEIGHT))
    if not -self._params.max_velocity <= state.puck_vel_x <= self._params.max_velocity:
      raise ValueError(_INVALID_VELOCITY.format(-self._params.max_velocity, self._params.max_velocity))
    if not -self._params.max_velocity <= state.puck_vel_y <= self._params.max_velocity:
      raise ValueError(_INVALID_VELOCITY.format(-self._params.max_velocity, self._params.max_velocity))

    self._state = copy.deepcopy(state)

    return self._get_observation()

  def render(self) -> np.ndarray:
    """Creates an image of the current environment state.

        Returns:
          A NumPy array giving an image of the environment state.
    """
    image = Image.new("RGB", (_IMAGE_WIDTH, _IMAGE_HEIGHT), "white")
    dct = ImageDraw.Draw(image)

    def get_circle_coordinates(x, y, radius):
      return [(x - radius) * _PIXELS_PER_SQ,
              (y - radius) * _PIXELS_PER_SQ,
              (x + radius) * _PIXELS_PER_SQ,
              (y + radius) * _PIXELS_PER_SQ]

    # Draw the goal
    dct.ellipse(
      get_circle_coordinates(self._state.goal_pos_x,
                             self._state.goal_pos_y,
                             _GOAL_RADIUS),
      fill=_RED_HEX)

    # Draw the puck
    dct.ellipse(
      get_circle_coordinates(self._state.puck_pos_x,
                             self._state.puck_pos_y,
                             _PUCK_RADIUS),
      fill=_GREEN_HEX)

    return np.asarray(image, dtype=np.uint8)
