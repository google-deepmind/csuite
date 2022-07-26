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

"""Wrapper for converting a csuite base.Environment to dm_env.Environment."""

from csuite.environments import base
import dm_env


class DMEnvFromCSuite(dm_env.Environment):
  """A wrapper to convert a CSuite environment to a dm_env.Environment."""

  def __init__(self, csuite_env: base.Environment):
    self._csuite_env = csuite_env

  def reset(self) -> dm_env.TimeStep:
    observation = self._csuite_env.start()
    return dm_env.restart(observation)

  def step(self, action) -> dm_env.TimeStep:
    # Convert the csuite step result to a dm_env TimeStep.
    observation, reward = self._csuite_env.step(action)
    return dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        observation=observation,
        reward=reward,
        discount=1.0)

  def observation_spec(self):
    return self._csuite_env.observation_spec()

  def action_spec(self):
    return self._csuite_env.action_spec()

  def get_state(self):
    return self._csuite_env.get_state()

  def set_state(self, state):
    self._csuite_env.set_state(state)

  def render(self):
    return self._csuite_env.render()
