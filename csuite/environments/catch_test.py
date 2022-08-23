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

"""Tests for catch."""

from absl.testing import absltest
from absl.testing import parameterized
from csuite.environments import catch


class CatchTest(parameterized.TestCase):

  def test_environment_setup(self):
    """Tests environment initialization."""
    env = catch.Catch()
    self.assertIsNotNone(env)

  def test_start(self):
    """Tests environment start."""
    env = catch.Catch()
    params = env.get_config()

    with self.subTest(name='step_without_start'):
      # Calling step before start should raise an error.
      with self.assertRaises(RuntimeError):
        env.step(catch.Action.LEFT)

    with self.subTest(name='start_state'):
      start_obs = env.start()
      state = env.get_state()
      # Paddle should be positioned at the bottom of the board.
      self.assertEqual(state.paddle_y, params.rows - 1)
      self.assertEqual(start_obs[state.paddle_y, state.paddle_x], 1)

      # First ball should be positioned at the top of the board.
      ball_x = state.balls[0][0]
      ball_y = state.balls[0][1]
      self.assertEqual(ball_y, 0)
      self.assertEqual(start_obs[ball_y, ball_x], 1)

  def test_invalid_state(self):
    """Tests setting environment state with invalid fields."""
    env = catch.Catch()
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
    env = catch.Catch()
    env.start()

    with self.subTest(name='invalid_action'):
      with self.assertRaises(ValueError):
        env.step(3)

    with self.subTest(name='move_left_step'):
      current_state = env.get_state()
      current_state.paddle_x = paddle_x
      env.set_state(current_state)

      env.step(catch.Action.LEFT)
      state = env.get_state()

      # Paddle x-position should have moved left by 1 unless at the edge.
      self.assertEqual(state.paddle_x, expected_left_x)

    with self.subTest(name='move_right_step'):
      current_state = env.get_state()
      current_state.paddle_x = paddle_x
      env.set_state(current_state)

      env.step(catch.Action.RIGHT)
      state = env.get_state()

      # Paddle x-position should have moved right by 1 unless at the edge.
      self.assertEqual(state.paddle_x, expected_right_x)

    with self.subTest(name='stay_step'):
      current_state = env.get_state()
      current_state.paddle_x = paddle_x
      env.set_state(current_state)

      env.step(catch.Action.STAY)
      state = env.get_state()
      self.assertEqual(state.paddle_x, paddle_x)

  def test_ball_hitting_bottom(self):
    """Tests environment updates when a ball hits the bottom of board."""
    env = catch.Catch()
    env.start()
    params = env.get_config()
    cur_state = env.get_state()

    with self.subTest(name='no_collision_with_paddle'):
      # Set environment state to immediately before ball falls to the bottom.
      cur_state.paddle_x = 0
      cur_state.paddle_y = params.rows - 1
      cur_state.balls = [(2, params.rows - 2)]
      env.set_state(cur_state)
      _, reward = env.step(catch.Action.STAY)

      # Reward returned should equal -1.
      self.assertEqual(reward, -1)

    with self.subTest(name='collision_with_paddle'):
      # Set environment state to immediately before ball falls to the bottom.
      cur_state.paddle_x = 2
      cur_state.paddle_y = params.rows - 1
      cur_state.balls = [(2, params.rows - 2)]
      env.set_state(cur_state)
      _, reward = env.step(catch.Action.STAY)

      # Reward returned should equal 1.
      self.assertEqual(reward, 1)

  def test_catching_one_ball_from_start(self):
    """Test running from environment start for the duration of one ball falling."""
    env = catch.Catch()
    env.start()
    params = env.get_config()
    cur_state = env.get_state()

    # Set environment state such that ball and paddle are horizontally centered
    # and the ball is at the top of the board.
    cur_state.paddle_x = 2
    cur_state.paddle_y = params.rows - 1
    cur_state.balls = [(2, 0)]
    env.set_state(cur_state)

    # For eight steps, alternate between moving left and right.
    for _ in range(4):
      # Here reward should equal 0.
      _, reward = env.step(catch.Action.RIGHT)
      self.assertEqual(reward, 0)
      _, reward = env.step(catch.Action.LEFT)
      self.assertEqual(reward, 0)

    # For the last step, choose to stay - ball should fall on paddle
    # and reward should equal 1.
    _, reward = env.step(catch.Action.STAY)
    self.assertEqual(reward, 1)

  def test_catching_two_balls_from_start(self):
    """Test running environment for the duration of two balls falling."""
    env = catch.Catch()
    env.start()
    params = env.get_config()
    cur_state = env.get_state()

    # Set environment state such that there are two balls at the top and the
    # second row of the board, and paddle is horizontally centered.
    cur_state.paddle_x = 2
    cur_state.paddle_y = params.rows - 1
    cur_state.balls = [(0, 1), (2, 0)]
    env.set_state(cur_state)

    # For eight steps, repeatedly move left - ball in second row should fall on
    # paddle.
    for _ in range(7):
      # Here reward should equal 0.
      _, reward = env.step(catch.Action.LEFT)
      self.assertEqual(reward, 0)

    _, reward = env.step(catch.Action.LEFT)
    self.assertEqual(reward, 1)
    # Now move right - the second ball should reach the bottom of the board
    # and the paddle should not catch it.
    _, reward = env.step(catch.Action.RIGHT)
    self.assertEqual(reward, -1)


if __name__ == '__main__':
  absltest.main()
