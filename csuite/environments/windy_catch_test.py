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

"""Tests for WindyCatch."""

from absl.testing import absltest
from absl.testing import parameterized
from csuite.environments import windy_catch


class WindyCatchTest(parameterized.TestCase):

  def test_environment_setup(self):
    """Tests environment initialization."""
    env = windy_catch.WindyCatch()
    self.assertIsNotNone(env)

  def test_start(self):
    """Tests environment start."""
    env = windy_catch.WindyCatch()
    params = env.get_config()

    with self.subTest(name='step_without_start'):
      # Calling step before start should raise an error.
      with self.assertRaises(RuntimeError):
        env.step(windy_catch.Action.LEFT)

    with self.subTest(name='start_state'):
      start_obs = env.start()
      state = env.get_state()
      # Paddle should be positioned at the bottom of the board.
      self.assertEqual(state.paddle_y, params.rows - 1)
      paddle_idx = state.paddle_y * params.columns + state.paddle_x
      self.assertEqual(start_obs[paddle_idx], 1)

      # First ball should be positioned at the top of the board.
      ball_x = state.balls[0][0]
      ball_y = state.balls[0][1]
      self.assertEqual(ball_y, 0)
      ball_idx = ball_y * params.columns + ball_x
      self.assertEqual(start_obs[ball_idx], 1)

  def test_invalid_state(self):
    """Tests setting environment state with invalid fields."""
    env = windy_catch.WindyCatch()
    env.start()

    with self.subTest(name='paddle_out_of_range'):
      new_state = env.get_state()
      new_state.paddle_x = 5
      with self.assertRaises(ValueError):
        env.set_state(new_state)

    with self.subTest(name='balls_out_of_range'):
      new_state = env.get_state()
      new_state.balls = [(0, -1)]
      with self.assertRaises(ValueError):
        env.set_state(new_state)

  @parameterized.parameters((0, 0, 1), (2, 1, 3), (4, 3, 4))
  def test_one_step(self, paddle_x, expected_left_x, expected_right_x):
    """Tests one environment step given the x-position of the paddle."""
    env = windy_catch.WindyCatch()
    env.start()

    with self.subTest(name='invalid_action'):
      with self.assertRaises(ValueError):
        env.step(3)

    with self.subTest(name='move_left_step'):
      current_state = env.get_state()
      current_state.paddle_x = paddle_x
      env.set_state(current_state)

      env.step(windy_catch.Action.LEFT)
      state = env.get_state()

      # Paddle x-position should have moved left by 1 unless at the edge.
      self.assertEqual(state.paddle_x, expected_left_x)

    with self.subTest(name='move_right_step'):
      current_state = env.get_state()
      current_state.paddle_x = paddle_x
      env.set_state(current_state)

      env.step(windy_catch.Action.RIGHT)
      state = env.get_state()

      # Paddle x-position should have moved right by 1 unless at the edge.
      self.assertEqual(state.paddle_x, expected_right_x)

    with self.subTest(name='stay_step'):
      current_state = env.get_state()
      current_state.paddle_x = paddle_x
      env.set_state(current_state)

      env.step(windy_catch.Action.STAY)
      state = env.get_state()
      self.assertEqual(state.paddle_x, paddle_x)

  def test_wind(self):
    """Tests the wind."""
    env = windy_catch.WindyCatch(spawn_probability=0.0)
    env.start()

    with self.subTest(name='wind_stay'):
      state = env.get_state()
      state.balls = [(0, 0), (2, 0), (4, 0)]
      state.wind_direction = [True, False, False]
      env.set_state(state)

      env.step(windy_catch.Action.STAY)

      b0, b1, b2 = env.get_state().balls
      self.assertEqual(b0[0], 0)
      self.assertEqual(b0[1], 1)
      self.assertEqual(b1[0], 2)
      self.assertEqual(b1[1], 1)
      self.assertEqual(b2[0], 4)
      self.assertEqual(b2[1], 1)

    with self.subTest(name='wind_left'):
      state = env.get_state()
      state.balls = [(0, 0), (2, 0), (4, 0)]
      state.wind_direction = [False, True, False]
      env.set_state(state)

      env.step(windy_catch.Action.STAY)

      b0, b1, b2 = env.get_state().balls
      self.assertEqual(b0[0], 4)
      self.assertEqual(b0[1], 1)
      self.assertEqual(b1[0], 1)
      self.assertEqual(b1[1], 1)
      self.assertEqual(b2[0], 3)
      self.assertEqual(b2[1], 1)

    with self.subTest(name='wind_right'):
      state = env.get_state()
      state.balls = [(0, 0), (2, 0), (4, 0)]
      state.wind_direction = [False, False, True]
      env.set_state(state)

      env.step(windy_catch.Action.STAY)

      b0, b1, b2 = env.get_state().balls
      self.assertEqual(b0[0], 1)
      self.assertEqual(b0[1], 1)
      self.assertEqual(b1[0], 3)
      self.assertEqual(b1[1], 1)
      self.assertEqual(b2[0], 0)
      self.assertEqual(b2[1], 1)


if __name__ == '__main__':
  absltest.main()
