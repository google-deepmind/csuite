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

"""Tests for access_control."""

from absl.testing import absltest
from absl.testing import parameterized
from csuite.environments import access_control


class AccessControlTest(parameterized.TestCase):

  def test_environment_setup(self):
    """Tests environment initialization."""
    access_control.AccessControl()

  def test_start(self):
    """Tests environment start."""
    env = access_control.AccessControl()
    params = env.get_config()

    with self.subTest(name='step_without_start'):
      # Calling step before start should raise an error.
      with self.assertRaises(RuntimeError):
        env.step(access_control.Action.REJECT)

    with self.subTest(name='start_state'):
      _ = env.start()
      state = env.get_state()
      self.assertEqual(state.num_busy_servers, 0)
      self.assertIn(state.incoming_priority, params.priorities)

  def test_invalid_state(self):
    """Tests setting environment state with invalid fields."""
    env = access_control.AccessControl()
    _ = env.start()
    cur_state = env.get_state()

    with self.subTest(name='invalid_state'):
      cur_state.num_busy_servers = 5
      cur_state.incoming_priority = -1
      with self.assertRaises(ValueError):
        env.set_state(cur_state)

    with self.subTest(name='invalid_priority'):
      cur_state.num_busy_servers = -1
      cur_state.incoming_priority = 8
      with self.assertRaises(ValueError):
        env.set_state(cur_state)

  @parameterized.parameters(0, 1, 9, 10)
  def test_one_step(self, new_num_busy_servers):
    """Tests environment step."""
    env = access_control.AccessControl()
    _ = env.start()
    params = env.get_config()

    with self.subTest(name='invalid_action'):
      with self.assertRaises(ValueError):
        env.step(5)

    with self.subTest(name='reject_step'):
      # Change the number of busy servers in the environment state.
      current_state = env.get_state()
      current_state.num_busy_servers = new_num_busy_servers
      env.set_state(current_state)

      next_obs, reward = env.step(access_control.Action.REJECT)
      state = env.get_state()
      # Next observation should give a valid state index, reward is zero, and
      # number of busy servers has not increased.
      self.assertIn(next_obs, range(env.num_states))
      self.assertEqual(reward, 0)
      self.assertLessEqual(state.num_busy_servers, new_num_busy_servers)

    with self.subTest(name='accept_step'):
      # Change current state to new number of busy servers.
      current_state = env.get_state()
      current_state.num_busy_servers = new_num_busy_servers
      env.set_state(current_state)

      next_obs, reward = env.step(access_control.Action.ACCEPT)
      state = env.get_state()

      if new_num_busy_servers == params.num_servers:  # all servers busy.
        # Reward is zero even if agent tries accepting, and number of busy
        # servers does not increase over the total available.
        self.assertEqual(reward, 0)
        self.assertLessEqual(state.num_busy_servers, new_num_busy_servers)
      else:
        # Reward is incoming priority, and the number of busy servers can
        # increase by one.
        self.assertIn(reward, params.priorities)
        self.assertLessEqual(state.num_busy_servers, new_num_busy_servers + 1)

  def test_runs_from_start(self):
    """Creates an environment and runs for 10 steps."""
    env = access_control.AccessControl()
    _ = env.start()

    for _ in range(10):
      _, _ = env.step(access_control.Action.ACCEPT)


if __name__ == '__main__':
  absltest.main()
