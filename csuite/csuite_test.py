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

from absl.testing import absltest
from absl.testing import parameterized

import csuite


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


if __name__ == "__main__":
  absltest.main()
