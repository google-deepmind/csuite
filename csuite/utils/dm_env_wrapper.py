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
        discount=None)

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
