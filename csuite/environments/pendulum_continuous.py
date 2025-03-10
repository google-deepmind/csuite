# Modified by Abhishek Naik in Oct 2024 
# to use a single continuous action that provides a torque to the pendulum
# in the range [-1, 1].

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

"""Implementation of a continuing Pendulum environment with continuous actions.

Environment description can be found in the `Pendulum' Environment
class.
"""

import copy
import dataclasses
import enum
from typing import Any, Callable, Optional

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
_TORQUE_SCALE_FACTOR = 1.0

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


@dataclasses.dataclass
class State:
  """State of a continuing pendulum environment.

  Attributes:
    angle: a float in [0, 2*pi] giving the angle in radians of the pendulum. An
      angle of 0 indicates that the pendulum is hanging downwards.
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
  torque_scale_factor: float


# Default start state and reward function.
def start_from_bottom():
  """Returns default start state with pendulum hanging vertically downwards."""
  return State(angle=0., velocity=0.)

def start_from_top():
   """Returns default start state with pendulum vertically upwards."""
   return State(angle=np.pi, velocity=0.)

def start_from_side():
 """Returns default start state with pendulum sideways."""
 return State(angle=np.pi/2, velocity=0.)

def start_close_to_bottom():
 """Returns default start state with pendulum almost vertically downwards."""
 return State(angle=np.pi/6, velocity=0.)

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

def dense_reward(state: State,
                 unused_torque: Any,
                 unused_step_size: Any,) -> float:
  """Returns a dense reward for the continuing pendulum problem.

  Args:
    state: A State object containing the current angle and velocity.

  Returns:
    A reward proportional to the angular distance of the pendulum
    from the goal position.
  """
  cost = (np.pi - _alias_angle(state.angle)) ** 2 + 0.1 * state.velocity ** 2
  return -cost

def _alias_angle(angle: float) -> float:
  """Returns an angle between 0 and 2*pi."""
  return angle % (2 * np.pi)


class PendulumContinuous(base.Environment):
  """A continuing pendulum environment.

  Starting from a hanging down position, swing up a single pendulum to the
  inverted position and maintain the pendulum in this position.

  Most of the default arguments model the pendulum described in "A Comparison of
  Direct and Model-Based Reinforcement Learning" (Atkeson & Santamaria, 1997).
  The key differences are the following:
  1) This environment is now continuing, i.e. the "trial length" is infinite.
  2) Instead of directly returning the angle of the pendulum in the observation,
  this environment returns the cosine and sine of the angle.
  3) There are only three discrete actions (apply a torque of -1, apply a torque
  of +1, and apply no-op) as opposed to a continuous torque value.
  4) The default reward function is implemented as a sparse reward, i.e. there
  is only a reward of 1 attained when the angle is in the region specified by
  the interval (pi - reward_angle, pi + reward_angle).

  The pendulum's motion is described by the equation
  ```
      theta'' = tau - mu * theta' - g * sin(theta),
  ```

  where theta is the angle, mu is the friction coefficient, tau is the torque,
  and g is the acceleration due to gravity.

  Observations are NumPy arrays of the form [cos(angle), sin(angle), velocity]
  where angle is in [0, 2*pi] and velocity is in [-max_speed, max_speed].
  """

  def __init__(self,
               start_state_fn=start_from_bottom,
               friction=_FRICTION,
               gravity=_GRAVITY,
               simulation_step_size=_SIMULATION_STEP_SIZE,
               act_step_period=_ACT_STEP_PERIOD,
               max_speed=_MAX_SPEED,
               reward_fn=dense_reward,
               torque_scale_factor=_TORQUE_SCALE_FACTOR,
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
      max_speed: A float giving the maximum speed (in radians/second) allowed in
        the simulation.
      reward_fn: A callable which returns a float reward given current state.
      seed: Seed for the internal random number generator.
      torque_scale_factor: scales the torque from -1/1 to some -k/k
    """
    del seed
    self._params = Params(
        start_state_fn=start_state_fn,
        friction=friction,
        gravity=gravity,
        simulation_step_size=simulation_step_size,
        act_step_period=act_step_period,
        max_speed=max_speed,
        reward_fn=reward_fn,
        torque_scale_factor=torque_scale_factor)
    self._state = None
    self._torque = 0

  def start(self, seed: Optional[int] = None):
    """Initializes the environment and returns an initial observation."""
    del seed
    self._state = self._params.start_state_fn()
    return np.array((np.cos(self._state.angle),
                     np.sin(self._state.angle),
                     self._state.velocity),
                    dtype=np.float32)

  @property
  def started(self):
    """True if the environment has been started, False otherwise."""
    # An unspecified state implies that the environment needs to be started.
    return self._state is not None

  def step(self, action):
    """Updates the environment state and returns an observation and reward.

    Args:
      action: an np.array with a single real number in [-1, 1] indicating the magnitude of torque.

    Returns:
      A tuple giving the next observation in the form of a NumPy array
      and the reward as a float.

    Raises:
      RuntimeError: If state has not yet been initialized by `start`.
    """
    # Check if state has been initialized.
    if not self.started:
      raise RuntimeError(base.STEP_WITHOUT_START_ERR)

    self._torque = action[0] * self._params.torque_scale_factor

    # Integrate over time steps to get new angle and velocity.
    new_angle = self._state.angle
    new_velocity = self._state.velocity

    for _ in range(self._params.act_step_period):
      new_velocity += ((self._torque - self._params.friction * new_velocity -
                        self._params.gravity * np.sin(new_angle)) *
                       self._params.simulation_step_size)
      new_angle += new_velocity * self._params.simulation_step_size

    # Ensure the angle is between 0 and 2*pi.
    new_angle = _alias_angle(new_angle)

    # Clip velocity to max_speed.
    new_velocity = np.clip(new_velocity, -self._params.max_speed,
                           self._params.max_speed)

    self._state = State(angle=new_angle, velocity=new_velocity)
    return (np.array((np.cos(self._state.angle),
                      np.sin(self._state.angle),
                      self._state.velocity),
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
    return specs.BoundedArray((1,), np.float64, minimum=-1.0, maximum=1.0)

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

    return np.array((np.cos(self._state.angle),
                     np.sin(self._state.angle),
                     self._state.velocity),
                    dtype=np.float32)

  def render(self):
    image = Image.new("RGB", (_IMAGE_SIZE, _IMAGE_SIZE), "white")
    dct = ImageDraw.Draw(image)
    # Get x and y positions of the pendulum tip relative to the center.
    x_pos = np.sin(self._state.angle)
    y_pos = np.cos(self._state.angle)

    def abs_coordinates(x, y, scale_factor=_SCALE_FACTOR):
      """Return absolute coordinates given coordinates relative to center."""
      return (x * _CENTER_IMAGE * scale_factor + _CENTER_IMAGE,
              y * _CENTER_IMAGE * scale_factor + _CENTER_IMAGE)

    torque_scale_factor = abs(self._torque)

    # Draw reward range region.
    boundary_x = _CENTER_IMAGE * (1 - _SCALE_FACTOR)
    pendulum_bounding_box = [(boundary_x, boundary_x),
                             (_IMAGE_SIZE - boundary_x,
                              _IMAGE_SIZE - boundary_x)]
    arrow_pos = _CENTER_IMAGE * (1 - torque_scale_factor)
    arrow_bounding_box = [(arrow_pos, arrow_pos),
                          (_IMAGE_SIZE - arrow_pos,
                           _IMAGE_SIZE - arrow_pos)]                              
    dct.pieslice(
        pendulum_bounding_box,
        start=(270 - _REWARD_ANGLE),
        end=(270 + _REWARD_ANGLE),
        fill=_LIGHT_GREEN)

    # Get absolute coordinates of the pendulum tip.
    tip_coords = abs_coordinates(x_pos, y_pos, _SCALE_FACTOR)
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
          arrow_bounding_box,
          start=360 - _TORQUE_ANGLE,
          end=_TORQUE_ANGLE,
          fill="blue",
          width=_ARROW_WIDTH)
      # Draw arrow heads.
      arrow_x, arrow_y = abs_coordinates(
          np.cos(_TORQUE_ANGLE * _RADIAN_MULTIPLIER),
          -np.sin(_TORQUE_ANGLE * _RADIAN_MULTIPLIER), torque_scale_factor)
      dct.regular_polygon((arrow_x, arrow_y, _ARROW_WIDTH * 1.5),
                          n_sides=3,
                          rotation=_TORQUE_ANGLE,
                          fill="blue")

    elif self._torque < 0:
      scale_factor = abs(self._torque)
      dct.arc(
          arrow_bounding_box,
          start=180 - _TORQUE_ANGLE,
          end=180 + _TORQUE_ANGLE,
          fill="blue",
          width=_ARROW_WIDTH)
      # Draw arrow heads.
      arrow_x, arrow_y = abs_coordinates(
          -np.cos(_TORQUE_ANGLE * _RADIAN_MULTIPLIER),
          -np.sin(_TORQUE_ANGLE * _RADIAN_MULTIPLIER), torque_scale_factor)
      dct.regular_polygon((arrow_x, arrow_y, _ARROW_WIDTH * 1.5),
                          n_sides=3,
                          rotation=-_TORQUE_ANGLE,
                          fill="blue")

    return np.asarray(image, dtype=np.uint8)
