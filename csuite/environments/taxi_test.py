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

"""Tests for taxi."""

from absl.testing import absltest
from absl.testing import parameterized
from csuite.environments import taxi


class TaxiTest(parameterized.TestCase):

    def test_environment_setup(self):
        """Tests environment initialization."""
        env = taxi.Taxi()
        self.assertIsNotNone(env)

    def test_start(self):
        """Tests environment start."""
        env = taxi.Taxi()

        with self.subTest(name='step_without_start'):
            # Calling step before start should raise an error.
            with self.assertRaises(RuntimeError):
                env.step(taxi.Action.NORTH)

        with self.subTest(name='start_state'):
            _ = env.start()
            state = env.get_state()
            self.assertIn(state.taxi_x, range(5))
            self.assertIn(state.taxi_y, range(5))
            # Original passenger location should not be in the taxi.
            self.assertIn(state.passenger_loc, range(4))
            self.assertIn(state.destination, range(4))

    @parameterized.parameters(
        (2, 2, taxi.Action.NORTH, True), (2, 0, taxi.Action.NORTH, False),
        (2, 2, taxi.Action.SOUTH, True), (2, 4, taxi.Action.SOUTH, False),
        (1, 4, taxi.Action.EAST, True), (1, 1, taxi.Action.EAST, False),
        (2, 4, taxi.Action.WEST, True), (2, 1, taxi.Action.WEST, False))
    def test_one_movement_step(self, x, y, action, can_move):
        """Tests one step with a movement action (North, East, South, West)."""
        env = taxi.Taxi()
        env.start()
        cur_state = env.get_state()

        # Create new state from input parameters and set environment to this state.
        cur_state.taxi_x = x
        cur_state.taxi_y = y
        cur_state.passenger_loc = 0
        cur_state.destination = 2
        env.set_state(cur_state)

        # Take movement step provided.
        env.step(action)
        next_state = env.get_state()
        if can_move:
            self.assertEqual(next_state.taxi_x,
                             cur_state.taxi_x + taxi.Action(action).dx)
            self.assertEqual(next_state.taxi_y,
                             cur_state.taxi_y + taxi.Action(action).dy)
        else:
            self.assertEqual(next_state.taxi_x, cur_state.taxi_x)
            self.assertEqual(next_state.taxi_y, cur_state.taxi_y)

    @parameterized.parameters((0, 0, 0, 2, taxi.Action.PICKUP, True),
                              (0, 1, 0, 2, taxi.Action.PICKUP, False),
                              (0, 1, 4, 2, taxi.Action.PICKUP, False),
                              (3, 4, 4, 3, taxi.Action.DROPOFF, True),
                              (2, 4, 4, 3, taxi.Action.DROPOFF, False),
                              (3, 4, 3, 3, taxi.Action.DROPOFF, False))
    def test_pickup_dropoff(self, x, y, pass_loc, dest, action, is_success):
        """Tests the two passenger actions (pickup and dropoff)."""
        env = taxi.Taxi()
        env.start()
        cur_state = env.get_state()

        # Create new state from input parameters and set environment to this state.
        cur_state.taxi_x = x
        cur_state.taxi_y = y
        cur_state.passenger_loc = pass_loc
        cur_state.destination = dest
        env.set_state(cur_state)
        _, reward = env.step(action)

        # Check correct reward: for successful dropoffs, reward is 20. For
        # successful pickups, reward is 0. For incorrect actions, reward is -10.
        if is_success and action == taxi.Action.DROPOFF:
            self.assertEqual(reward, 20)
        elif is_success:
            self.assertEqual(reward, 0)
        else:
            self.assertEqual(reward, -10)

        if is_success and action == taxi.Action.PICKUP:
            # Check passenger is in the taxi.
            next_state = env.get_state()
            self.assertEqual(next_state.passenger_loc, 4)

    def test_runs_from_start(self):
        """Tests running a full passenger pickup and dropoff sequence."""
        env = taxi.Taxi()
        env.start()
        cur_state = env.get_state()

        # Set state to have passenger and taxi on the red square, and destination
        # on the blue square.
        cur_state.taxi_x = 0
        cur_state.taxi_y = 0
        cur_state.passenger_loc = 0
        cur_state.destination = 3

        env.set_state(cur_state)
        # Pick up the passenger.
        env.step(taxi.Action.PICKUP)

        for _ in range(2):
            _, reward = env.step(taxi.Action.SOUTH)
            self.assertEqual(reward, 0)
        for _ in range(3):
            _, reward = env.step(taxi.Action.EAST)
            self.assertEqual(reward, 0)
        for _ in range(2):
            _, reward = env.step(taxi.Action.SOUTH)
            self.assertEqual(reward, 0)

        # Drop off the passenger.
        _, reward = env.step(taxi.Action.DROPOFF)
        self.assertEqual(reward, 20)


if __name__ == '__main__':
    absltest.main()
