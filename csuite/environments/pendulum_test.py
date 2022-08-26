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

"""Tests for pendulum."""

from absl.testing import absltest
from csuite.environments import pendulum


class PendulumTest(absltest.TestCase):

  def test_environment_setup(self):
    """Tests environment initialization."""
    env = pendulum.Pendulum()
    self.assertIsNotNone(env)

  def test_start(self):
    """Tests environment start."""
    env = pendulum.Pendulum()

    with self.subTest(name='step_without_start'):
      # Calling step before start should raise an error.
      with self.assertRaises(RuntimeError):
        env.step(pendulum.Action.NEGATIVE)

    with self.subTest(name='start_state'):
      start_obs = env.start()
      # Initial cosine of the angle should be 1.
      # Initial sine of the angle and initial velocity should be 0.
      self.assertEqual(start_obs[0], 1.)
      self.assertEqual(start_obs[1], 0.)
      self.assertEqual(start_obs[2], 0.)

  def test_one_step(self):
    """Tests one environment step."""
    env = pendulum.Pendulum()
    env.start()
    _, reward = env.step(pendulum.Action.NEGATIVE)
    self.assertEqual(reward, 0.)
    _, reward = env.step(pendulum.Action.POSITIVE)
    self.assertEqual(reward, 0.)
    _, reward = env.step(pendulum.Action.STAY)
    self.assertEqual(reward, 0.)

  def test_setting_state(self):
    """Tests setting environment state and solver."""
    env = pendulum.Pendulum()
    old_obs = env.start()
    # Take two steps adding +1 torque, then set state to downwards position.
    for _ in range(2):
      old_obs, _ = env.step(pendulum.Action.POSITIVE)
    new_state = pendulum.State(angle=0., velocity=0.)
    new_obs = env.set_state(new_state)
    for _ in range(2):
      new_obs, _ = env.step(pendulum.Action.POSITIVE)

    # If the solver was properly updated, the two observations are the same.
    self.assertLessEqual(abs(new_obs[0]), old_obs[0])
    self.assertLessEqual(abs(new_obs[1]), old_obs[1])
    self.assertLessEqual(abs(new_obs[2]), old_obs[2])


if __name__ == '__main__':
  absltest.main()
