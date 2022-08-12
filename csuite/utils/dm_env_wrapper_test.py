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

"""Test for DMEnvFromCSuite."""

from absl.testing import absltest
import csuite
from dm_env import test_utils


class DMEnvFromCSuiteTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    csuite_env = csuite.load('catch')
    return csuite.dm_env_wrapper.DMEnvFromCSuite(csuite_env)


if __name__ == '__main__':
  absltest.main()

