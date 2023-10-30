"""Tests for PuckWorld."""

from absl.testing import absltest
from csuite.environments.experimental import puckworld


class PuckWorldTest(absltest.TestCase):

  def test_environment_setup(self):
    """Tests environment initialization."""
    env = puckworld.PuckWorld()
    self.assertIsNotNone(env)

  def test_start(self):
    """Tests environment start."""
    env = puckworld.PuckWorld()

    with self.subTest(name='step_without_start'):
      # Calling step before start should raise an error.
      with self.assertRaises(RuntimeError):
        env.step(puckworld.Action.NORTH)

    with self.subTest(name='start_state'):
      start_obs = env.start()
      # Initial positions should be in the expected range.
      # Initial velocity of puck should be zero
      self.assertBetween(start_obs[0], 0., 5.)
      self.assertBetween(start_obs[1], 0., 5.)
      self.assertEqual(start_obs[2], 0.)
      self.assertEqual(start_obs[3], 0.)
      self.assertBetween(start_obs[4], 0., 5.)
      self.assertBetween(start_obs[5], 0., 5.)

  def test_goal_update(self):
    env = puckworld.PuckWorld(goal_update_interval=10)
    first_obs = env.start()
    total_test_steps = 15
    first_test_after = 5

    obs = None
    for _ in range(first_test_after):
      obs, _ = env.step(puckworld.Action.NORTH)

    with self.subTest(name='goal_is_the_same'):
      self.assertEqual(obs[4], first_obs[4])
      self.assertEqual(obs[5], first_obs[5])

    for _ in range(total_test_steps - first_test_after):
      obs, _ = env.step(puckworld.Action.NORTH)

    with self.subTest(name='goal_has_updated'):
      self.assertNotEqual(obs[4], first_obs[4])
      self.assertNotEqual(obs[5], first_obs[5])

  def test_setting_state(self):
    """Tests setting environment state."""
    env = puckworld.PuckWorld()
    old_obs = env.start()
    old_state = env.get_state()
    # Take two steps adding +1 torque, then set state to downwards position.
    for _ in range(2):
      old_obs, _ = env.step(puckworld.Action.SOUTH)
    new_obs = env.set_state(old_state)
    for _ in range(2):
      new_obs, _ = env.step(puckworld.Action.SOUTH)

    # If the state was properly updated, the two observations are the same.
    self.assertLessEqual(abs(new_obs[0]), old_obs[0])
    self.assertLessEqual(abs(new_obs[1]), old_obs[1])
    self.assertLessEqual(abs(new_obs[2]), old_obs[2])
    self.assertLessEqual(abs(new_obs[3]), old_obs[3])
    self.assertLessEqual(abs(new_obs[4]), old_obs[4])
    self.assertLessEqual(abs(new_obs[5]), old_obs[5])


if __name__ == '__main__':
  absltest.main()
