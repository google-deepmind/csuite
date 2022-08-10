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

"""Implementation of a continuing Pendulum environment with discrete actions.

Environment description can be found in the `Pendulum' Environment
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

# Converter for degrees to radians.
_RADIAN_MULTIPLIER = np.pi / 180

# Error messages.
_STEP_WITHOUT_START = ("Environment state has not been initialized."
                       "`start` must be called before calling `step`.")
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
  """Actions for the Pendulum environment.

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
  """State of a continuing pendulum environment.

  Attributes:
    angle: a float in [0, 2*pi] giving the angle in radians of the pendulum.
      An angle of 0 indicates that the pendulum is hanging downwards.
    velocity: a float in [-max_speed, max_speed] giving the angular velocity.
  """
  angle: float
  velocity: float


@dataclasses.dataclass
class Params:
  """Parameters of a continuing pendulum environment."""
  start_state_fn: Callable[[], State]
  friction: float
  gravity: float
  simulation_step_size: float
  act_step_period: int
  max_speed: float
  reward_fn: Callable[..., float]


# Default start state and reward function.
def start_from_bottom():
  """Returns default start state with pendulum hanging vertically downwards."""
  return State(angle=0., velocity=0.)


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
  if (np.pi - reward_angle_radians < state.angle
      < np.pi + reward_angle_radians):
    return 1.
  else:
    return 0.


# Dense reward implemented from Atkeson and Santamaria (1997).
# TODO(b/240199026): Decide whether we want to include dense reward.
def dense_reward(state: State, torque: float, stepsize: float) -> float:
  """Returns a dense reward for the continuing pendulum problem.

  As defined in "A Comparison of Direct and Model-Based Reinforcement
  Learning" (Atkeson & Santamaria, 1997), the reward is calculated as
  ```
    -((theta - pi) ^ 2 + torque ^ 2) * stepsize,
  ```
  where the theta is the angle in [0, 2pi], torque is the torque currently
  being applied, and stepsize is the current integration stepsize. The maximum
  reward achievable is 0 (when the pendulum is upright with zero torque applied)
  and the minimum reward achievable is -(pi^2 + 1) * 0.05 = -0.54348...

  Args:
    state: A State object containing the current angle and velocity.
    torque: A float giving the current motor torque.
    stepsize: A float giving the integration time step.

  Returns:
    A float giving the dense reward.
  """
  return -((state.angle - np.pi) ** 2 + torque ** 2) * stepsize


def _alias_angle(angle: float) -> float:
  """Returns an angle between 0 and 2*pi."""
  return angle % (2 * np.pi)


# TODO(b/240199204): Update docstring with different versions of pendulum.
class Pendulum(base.Environment):
  """A continuing pendulum environment.

  Starting from a hanging down position, swing up a single pendulum to the
  inverted position and maintain the pendulum in this position.

  Most of the default arguments model the pendulum described in "A Comparison of
  Direct and Model-Based Reinforcement Learning" (Atkeson & Santamaria, 1997).
  The key differences are the following:
  1) This environment is now continuing, i.e. the "trial length" is infinite.
  2) There are only three discrete actions (apply a torque of -1, apply a torque
  of +1, and apply no-op) as opposed to a continuous torque value.
  3) The default reward function is implemented as a sparse reward, i.e. there
  is only a reward of 1 attained when the angle is in the region specified by
  the interval (pi - reward_angle, pi + reward_angle).

  The pendulum's motion is described by the equation
  ```
      theta'' = tau - mu * theta' - g * sin(theta),
  ```

  where theta is the angle, mu is the friction coefficient, tau is the torque,
  and g is the acceleration due to gravity.

  Observations are NumPy arrays of the form of (angle, velocity) where angle
  is in [0, 2*pi] and velocity is in [-max_speed, max_speed].
  """

  def __init__(self,
               start_state_fn=start_from_bottom,
               friction=_FRICTION,
               gravity=_GRAVITY,
               simulation_step_size=_SIMULATION_STEP_SIZE,
               act_step_period=_ACT_STEP_PERIOD,
               max_speed=_MAX_SPEED,
               reward_fn=sparse_reward,
               seed=None):
    """Initializes a new pendulum environment.

    Args:
      start_state_fn: A callable which returns a `State` object giving the
        initial state.
      friction: A positive float giving the coefficient of friction.
      gravity: A float giving the acceleration due to gravity.
      simulation_step_size: The step size (in seconds) of the simulation.
      act_step_period: An integer giving the number of simulation steps for each
        action input.
      max_speed: A float giving the maximum speed (in radians/second) allowed
        in the simulation.
      reward_fn: A callable which returns a float reward given current state.
      seed: Seed for the internal random number generator.
    """
    self._params = Params(
        start_state_fn=start_state_fn,
        friction=friction,
        gravity=gravity,
        simulation_step_size=simulation_step_size,
        act_step_period=act_step_period,
        max_speed=max_speed,
        reward_fn=reward_fn)
    self._state = None
    self._torque = 0

  def start(self):
    """Initializes the environment and returns an initial observation."""
    self._state = self._params.start_state_fn()
    return np.array([self._state.angle, self._state.velocity], dtype=np.float32)

  def step(self, action):
    """Updates the environment state and returns an observation and reward.

    Args:
      action: An integer in {0, 1, 2} indicating whether to subtract one unit
        of torque, do nothing, or add one unit of torque.

    Returns:
      A tuple giving the next observation in the form of a NumPy array
      and the reward as a float.

    Raises:
      RuntimeError: If state has not yet been initialized by `start`.
    """
    # Check if state has been initialized.
    if self._state is None:
      raise RuntimeError(_STEP_WITHOUT_START)

    self._torque = Action(action).tau

    # Integrate over time steps to get new angle and velocity.
    new_angle = self._state.angle
    new_velocity = self._state.velocity

    for _ in range(self._params.act_step_period):
      new_velocity += (
          (self._torque  -
           self._params.friction * new_velocity -
           self._params.gravity * np.sin(new_angle))
          * self._params.simulation_step_size
      )
      new_angle += new_velocity * self._params.simulation_step_size

    # Ensure the angle is between 0 and 2*pi.
    new_angle = _alias_angle(new_angle)

    # Clip velocity to max_speed.
    new_velocity = np.clip(new_velocity, -self._params.max_speed,
                           self._params.max_speed)

    self._state = State(angle=new_angle, velocity=new_velocity)
    return (
        np.array([self._state.angle, self._state.velocity], dtype=np.float32),
        self._params.reward_fn(self._state,
                               self._torque,
                               self._params.simulation_step_size)
    )

  def observation_spec(self):
    """Describes the observation specs of the environment."""
    return specs.BoundedArray((2,), dtype=np.float32,
                              minimum=[0, -self._params.max_speed],
                              maximum=[2 * np.pi, self._params.max_speed])

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

    return np.array([self._state.angle, self._state.velocity])

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
    pendulum_bounding_box = [
        (boundary_x, boundary_x),
        (_IMAGE_SIZE - boundary_x, _IMAGE_SIZE - boundary_x)
    ]
    dct.pieslice(pendulum_bounding_box,
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
      dct.arc(pendulum_bounding_box,
              start=360 - _TORQUE_ANGLE, end=_TORQUE_ANGLE,
              fill="blue", width=_ARROW_WIDTH)
      # Draw arrow heads.
      arrow_x, arrow_y = abs_coordinates(
          np.cos(_TORQUE_ANGLE * _RADIAN_MULTIPLIER),
          -np.sin(_TORQUE_ANGLE * _RADIAN_MULTIPLIER)
      )
      dct.regular_polygon((arrow_x, arrow_y, _ARROW_WIDTH * 1.5),
                          n_sides=3, rotation=_TORQUE_ANGLE, fill="blue")

    elif self._torque < 0:
      dct.arc(pendulum_bounding_box,
              start=180 - _TORQUE_ANGLE, end=180 + _TORQUE_ANGLE,
              fill="blue", width=_ARROW_WIDTH)
      # Draw arrow heads.
      arrow_x, arrow_y = abs_coordinates(
          -np.cos(_TORQUE_ANGLE * _RADIAN_MULTIPLIER),
          -np.sin(_TORQUE_ANGLE * _RADIAN_MULTIPLIER)
      )
      dct.regular_polygon((arrow_x, arrow_y, _ARROW_WIDTH * 1.5),
                          n_sides=3, rotation=-_TORQUE_ANGLE, fill="blue")

    return np.asarray(image, dtype=np.uint8)
