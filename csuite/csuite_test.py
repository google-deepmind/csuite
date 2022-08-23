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

"""Tests all environments through the csuite interface."""

import typing

from absl.testing import absltest
from absl.testing import parameterized

import csuite
from dm_env import specs
import numpy as np


class CSuiteTest(parameterized.TestCase):

  @parameterized.parameters([e.value for e in csuite.EnvName])
  def test_envs(self, env_name):
    """Tests that we can use the environment in typical ways."""
    env = csuite.load(env_name)
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    obs = env.start()
    env.render()
    init_state = env.get_state()

    for i in range(2):
      with self.subTest(name="steps-render", step=i):
        env.render()
      with self.subTest(name="steps-observation_spec", step=i):
        observation_spec.validate(obs)
      with self.subTest(name="steps-step", step=i):
        obs, unused_reward = env.step(action_spec.generate_value())

    env.set_state(init_state)

  @parameterized.parameters([e.value for e in csuite.EnvName])
  def test_env_state_resets(self, env_name):
    """Tests that `get`ing and `set`ing state results in reproducibility."""
    # Since each environment is different, we employ a generic strategy that
    # should
    #
    # a) get us to a variety of states to query the state on,
    # b) take a number of steps from that state to check reproducibility.
    #
    # See the implementation for the specific strategy taken.
    num_steps_to_check = 4
    env = csuite.load(env_name)
    env.start()
    action_spec = env.action_spec()

    if not isinstance(action_spec, specs.DiscreteArray):
      raise NotImplementedError(
          "This test only supports environments with discrete action "
          "spaces for now. Please raise an issue if you want to work with a "
          "a non-discrete action space.")

    action_spec = typing.cast(specs.DiscreteArray, action_spec)
    for action in range(action_spec.num_values):
      env.step(action)
      orig_state = env.get_state()
      outputs_1 = [env.step(action) for _ in range(num_steps_to_check)]
      observations_1, rewards_1 = zip(*outputs_1)
      env.set_state(orig_state)
      outputs_2 = [env.step(action) for _ in range(num_steps_to_check)]
      observations_2, rewards_2 = zip(*outputs_2)
      with self.subTest("observations", action=action):
        self.assertSameObsSequence(observations_1, observations_2)
      with self.subTest("rewards", action=action):
        self.assertSequenceEqual(rewards_1, rewards_2)

  def assertSameObsSequence(self, seq1, seq2):
    """The observations are expected to be numpy objects."""
    # self.assertSameStructure(seq1, seq2)
    problems = []  # (idx, problem str)
    for idx, (el1, el2) in enumerate(zip(seq1, seq2)):
      try:
        np.testing.assert_array_equal(el1, el2)
      except AssertionError as e:
        problems.append((idx, str(e)))

    if problems:
      self.fail(
          f"The observation sequences (of length {len(seq1)}) are not the "
          "same. The differences are:\n" +
          "\n".join([f"at idx={idx}: {msg}" for idx, msg in problems]))


if __name__ == "__main__":
  absltest.main()
