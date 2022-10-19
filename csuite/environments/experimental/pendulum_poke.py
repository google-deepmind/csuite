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

"""Continuing pendulum with random perturbations in the reward region.

Environment description can be found in the `PendulumPoke' Environment
class.
"""

import copy
import dataclasses
import enum
from typing import Any, Callable
from csuite.environments import base
from dm_env import specs

import numpy as np
from PIL import Image
from PIL import ImageDraw

# Default environment variables.
_NUM_ACTIONS = 3  # Size of action space discretization.
_FRICTION = 0.1
_GRAVITY = 9.81
_SIMULATION_STEP_SIZE = 0.05
_ACT_STEP_PERIOD = 4
_MAX_SPEED = np.inf
_REWARD_ANGLE = 30
# Default environment variables for adding perturbations.
_PERTURB_PROB = 0.01
# TODO(b/243969989): Decide appropriate amount of perturbation torque.
_PERTURB_TORQUE = 10.

# Converter for degrees to radians.
_RADIAN_MULTIPLIER = np.pi / 180

# Error messages.
_INVALID_ANGLE = ("Invalid state: expected angle to be in range [0, 2pi].")

# Variables for pixel visualization of the environment.
_IMAGE_SIZE = 256
_CENTER_IMAGE = _IMAGE_SIZE // 2 - 1
_SCALE_FACTOR = 0.75
_PENDULUM_WIDTH = _IMAGE_SIZE // 64
_TIP_RADIUS = _IMAGE_SIZE // 24
_LIGHT_GREEN = "#d4ffd6"  # For shading the reward region.
_ARROW_WIDTH = _IMAGE_SIZE // 44
_TORQUE_ANGLE = 20


class Action(enum.IntEnum):
  """Actions for the PendulumPoke environment.

  There are three actions:
  0: Apply -1 torque.
  1: Do nothing.
  2: Apply +1 torque.
  """
  NEGATIVE, STAY, POSITIVE = range(3)

  @property
  def tau(self):
    """Maps NEGATIVE to -1, STAY to 0, and POSITIVE to 1."""
    return self.value - 1


@dataclasses.dataclass
class State:
  """State of a PendulumPoke environment.

  Attributes:
    angle: a float in [0, 2*pi] giving the angle in radians of the pendulum. An
      angle of 0 indicates that the pendulum is hanging downwards.
    velocity: a float in [-max_speed, max_speed] giving the angular velocity.
    rng: Internal NumPy pseudo-random number generator, included here for
      reproducibility purposes.
  """
  angle: float
  velocity: float
  rng: np.random.RandomState


@dataclasses.dataclass
class Params:
  """Parameters of a PendulumPoke environment."""
  friction: float
  gravity: float
  simulation_step_size: float
  act_step_period: int
  max_speed: float
  reward_fn: Callable[..., float]
  perturb_prob: float
  perturb_torque: float


def sparse_reward(state: State,
                  unused_torque: Any,
                  unused_step_size: Any,
                  reward_angle: int = _REWARD_ANGLE) -> float:
  """Returns a sparse reward for the continuing pendulum problem.

  Args:
    state: A State object containing the current angle and velocity.
    reward_angle: An integer denoting the angle from vertical, in degrees, where
      the pendulum is rewarding.

  Returns:
    A reward of 1 if the angle of the pendulum is within the range, and
    a reward of 0 otherwise.
  """
  reward_angle_radians = reward_angle * _RADIAN_MULTIPLIER
  if (np.pi - reward_angle_radians < state.angle <
      np.pi + reward_angle_radians):
    return 1.
  else:
    return 0.


def _alias_angle(angle: float) -> float:
  """Returns an angle between 0 and 2*pi."""
  return angle % (2 * np.pi)


class PendulumPoke(base.Environment):
  """A pendulum environment with a random addition of force in the reward region.

  This environment has the same parameters as the `Pendulum` environment, and
  additionally with a small probability, a force is applied to the pendulum
  if it is in the rewarding region to knock the pendulum over. The magnitude
  of the force is constant, and the direction that the force is applied is
  chosen uniformly at random.
  """

  def __init__(self,
               friction=_FRICTION,
               gravity=_GRAVITY,
               simulation_step_size=_SIMULATION_STEP_SIZE,
               act_step_period=_ACT_STEP_PERIOD,
               max_speed=_MAX_SPEED,
               reward_fn=sparse_reward,
               perturb_prob=_PERTURB_PROB,
               perturb_torque=_PERTURB_TORQUE,
               seed=None):
    """Initializes a new pendulum environment with random perturbations in the rewarding region.

    Args:
      friction: A positive float giving the coefficient of friction.
      gravity: A float giving the acceleration due to gravity.
      simulation_step_size: The step size (in seconds) of the simulation.
      act_step_period: An integer giving the number of simulation steps for each
        action input.
      max_speed: A float giving the maximum speed (in radians/second) allowed in
        the simulation.
      reward_fn: A callable which returns a float reward given current state.
      perturb_prob: A float giving the probability that a random force is
        applied to the pendulum if it is in the rewarding region.
      perturb_torque: A float giving the magnitude of the random force.
      seed: Seed for the internal random number generator.
    """
    self._params = Params(
        friction=friction,
        gravity=gravity,
        simulation_step_size=simulation_step_size,
        act_step_period=act_step_period,
        max_speed=max_speed,
        reward_fn=reward_fn,
        perturb_prob=perturb_prob,
        perturb_torque=perturb_torque)
    self._seed = seed
    self._state = None
    self._torque = 0
    self._perturb_direction = 0  # For visualization purposes.

  def start(self, seed=None):
    """Initializes the environment and returns an initial observation."""
    self._state = State(
        angle=0.,
        velocity=0.,
        rng=np.random.RandomState(self._seed if seed is None else seed))
    return np.array((np.cos(self._state.angle), np.sin(
        self._state.angle), self._state.velocity),
                    dtype=np.float32)

  @property
  def started(self):
    """True if the environment has been started, False otherwise."""
    # An unspecified state implies that the environment needs to be started.
    return self._state is not None

  def step(self, action):
    """Updates the environment state and returns an observation and reward.

    Args:
      action: An integer in {0, 1, 2} indicating whether to subtract one unit of
        torque, do nothing, or add one unit of torque.

    Returns:
      A tuple giving the next observation in the form of a NumPy array
      and the reward as a float.

    Raises:
      RuntimeError: If state has not yet been initialized by `start`.
    """
    # Check if state has been initialized.
    if not self.started:
      raise RuntimeError(base.STEP_WITHOUT_START_ERR)

    self._torque = Action(action).tau

    # Integrate over time steps to get new angle and velocity.
    new_angle = self._state.angle
    new_velocity = self._state.velocity

    # If the pendulum is in the rewarding region, with the given probability
    # add a force in a direction chosen uniformly at random.
    reward_angle_rad = _REWARD_ANGLE * _RADIAN_MULTIPLIER
    if ((np.pi - reward_angle_rad < new_angle < np.pi + reward_angle_rad) and
        self._state.rng.uniform() < self._params.perturb_prob):
      if self._state.rng.uniform() < 0.5:
        applied_torque = self._torque - self._params.perturb_torque
        self._perturb_direction = -1
      else:
        applied_torque = self._torque + self._params.perturb_torque
        self._perturb_direction = 1
    else:
      applied_torque = self._torque
      self._perturb_direction = 0

    for _ in range(self._params.act_step_period):
      new_velocity += ((applied_torque - self._params.friction * new_velocity -
                        self._params.gravity * np.sin(new_angle)) *
                       self._params.simulation_step_size)
      new_angle += new_velocity * self._params.simulation_step_size

    # Ensure the angle is between 0 and 2*pi.
    new_angle = _alias_angle(new_angle)

    # Clip velocity to max_speed.
    new_velocity = np.clip(new_velocity, -self._params.max_speed,
                           self._params.max_speed)

    self._state = State(
        angle=new_angle, velocity=new_velocity, rng=self._state.rng)
    return (np.array((np.cos(self._state.angle), np.sin(
        self._state.angle), self._state.velocity),
                     dtype=np.float32),
            self._params.reward_fn(self._state, self._torque,
                                   self._params.simulation_step_size))

  def observation_spec(self):
    """Describes the observation specs of the environment."""
    return specs.BoundedArray((3,),
                              dtype=np.float32,
                              minimum=[-1, -1, -self._params.max_speed],
                              maximum=[1, 1, self._params.max_speed])

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

    Returns:
      A NumPy array for the observation including the angle and velocity.
    """
    # Check that input state values are valid.
    if not 0 <= state.angle <= 2 * np.pi:
      raise ValueError(_INVALID_ANGLE)

    self._state = copy.deepcopy(state)

    return np.array((np.cos(self._state.angle), np.sin(
        self._state.angle), self._state.velocity),
                    dtype=np.float32)

  def render(self):
    image = Image.new("RGB", (_IMAGE_SIZE, _IMAGE_SIZE), "white")
    dct = ImageDraw.Draw(image)
    # Get x and y positions of the pendulum tip relative to the center.
    x_pos = np.sin(self._state.angle)
    y_pos = np.cos(self._state.angle)

    def abs_coordinates(x, y):
      """Return absolute coordinates given coordinates relative to center."""
      return (x * _SCALE_FACTOR * _CENTER_IMAGE + _CENTER_IMAGE,
              y * _SCALE_FACTOR * _CENTER_IMAGE + _CENTER_IMAGE)

    # Draw reward range region.
    boundary_x = _CENTER_IMAGE * (1 - _SCALE_FACTOR)
    pendulum_bounding_box = [(boundary_x, boundary_x),
                             (_IMAGE_SIZE - boundary_x,
                              _IMAGE_SIZE - boundary_x)]
    dct.pieslice(
        pendulum_bounding_box,
        start=(270 - _REWARD_ANGLE),
        end=(270 + _REWARD_ANGLE),
        fill=_LIGHT_GREEN)

    # Get absolute coordinates of the pendulum tip.
    tip_coords = abs_coordinates(x_pos, y_pos)
    # Draw pendulum line.
    dct.line([(_CENTER_IMAGE, _CENTER_IMAGE), tip_coords],
             fill="black",
             width=_PENDULUM_WIDTH)
    # Draw circular pendulum tip.
    x, y = tip_coords
    tip_bounding_box = [(x - _TIP_RADIUS, y - _TIP_RADIUS),
                        (x + _TIP_RADIUS, y + _TIP_RADIUS)]
    dct.ellipse(tip_bounding_box, fill="red")

    # Draw torque arrow.
    if self._torque > 0:
      dct.arc(
          pendulum_bounding_box,
          start=360 - _TORQUE_ANGLE,
          end=_TORQUE_ANGLE,
          fill="blue",
          width=_ARROW_WIDTH)
      # Draw arrow heads.
      arrow_x, arrow_y = abs_coordinates(
          np.cos(_TORQUE_ANGLE * _RADIAN_MULTIPLIER),
          -np.sin(_TORQUE_ANGLE * _RADIAN_MULTIPLIER))
      dct.regular_polygon((arrow_x, arrow_y, _ARROW_WIDTH * 1.5),
                          n_sides=3,
                          rotation=_TORQUE_ANGLE,
                          fill="blue")

    elif self._torque < 0:
      dct.arc(
          pendulum_bounding_box,
          start=180 - _TORQUE_ANGLE,
          end=180 + _TORQUE_ANGLE,
          fill="blue",
          width=_ARROW_WIDTH)
      # Draw arrow heads.
      arrow_x, arrow_y = abs_coordinates(
          -np.cos(_TORQUE_ANGLE * _RADIAN_MULTIPLIER),
          -np.sin(_TORQUE_ANGLE * _RADIAN_MULTIPLIER))
      dct.regular_polygon((arrow_x, arrow_y, _ARROW_WIDTH * 1.5),
                          n_sides=3,
                          rotation=-_TORQUE_ANGLE,
                          fill="blue")

    # Drawing perturbation arrow.
    if self._perturb_direction == 1:
      tip_coords = (_IMAGE_SIZE // 4 - _TIP_RADIUS, _IMAGE_SIZE // 8)
      arrow_coords = [
          tip_coords, (_IMAGE_SIZE // 4 + _TIP_RADIUS, _IMAGE_SIZE // 8)
      ]
      dct.line(arrow_coords, fill="red", width=_PENDULUM_WIDTH)
      dct.regular_polygon((tip_coords, _ARROW_WIDTH * 1.2),
                          rotation=90,
                          n_sides=3,
                          fill="red")
    elif self._perturb_direction == -1:
      tip_coords = (_IMAGE_SIZE * 3 // 4 + _TIP_RADIUS, _IMAGE_SIZE // 8)
      arrow_coords = [(_IMAGE_SIZE * 3 // 4 - _TIP_RADIUS, _IMAGE_SIZE // 8),
                      tip_coords]
      dct.line(arrow_coords, fill="red", width=_PENDULUM_WIDTH)
      dct.regular_polygon((tip_coords, _ARROW_WIDTH * 1.2),
                          rotation=270,
                          n_sides=3,
                          fill="red")
    return np.asarray(image, dtype=np.uint8)
