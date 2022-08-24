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

"""Helper(s) to load csuite environments."""

import enum
from typing import Dict, Optional, Union

from csuite.environments import access_control
from csuite.environments import catch
from csuite.environments import dancing_catch
from csuite.environments import pendulum
from csuite.environments import taxi
from csuite.environments import windy_catch
from csuite.utils import dm_env_wrapper
from csuite.utils import gym_wrapper


class EnvName(enum.Enum):
  ACCESS_CONTROL = 'access_control'
  CATCH = 'catch'
  DANCING_CATCH = 'dancing_catch'
  PENDULUM = 'pendulum'
  TAXI = 'taxi'
  WINDY_CATCH = 'windy_catch'


_ENVS = {
    EnvName.ACCESS_CONTROL: access_control.AccessControl,
    EnvName.CATCH: catch.Catch,
    EnvName.DANCING_CATCH: dancing_catch.DancingCatch,
    EnvName.WINDY_CATCH: windy_catch.WindyCatch,
    EnvName.TAXI: taxi.Taxi,
    EnvName.PENDULUM: pendulum.Pendulum,
}


def load(name: Union[EnvName, str],
         settings: Optional[Dict[str, Union[float, int, bool]]] = None):
  """Loads a csuite environment.

  Args:
    name: The enum or string specifying the environment name.
    settings: Optional `dict` of keyword arguments for the environment.

  Returns:
    An instance of the requested environment.
  """
  name = EnvName(name)
  settings = settings or {}
  return _ENVS[name](**settings)
