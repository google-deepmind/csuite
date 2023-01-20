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

"""Common utility functions."""

import numpy as np


def binary_board_to_rgb(board: np.ndarray) -> np.ndarray:
  """Converts a binary 2D array to an rgb array."""
  board = board.astype(np.uint8) * 255
  board = np.expand_dims(board, -1)
  board = np.tile(board, (1, 1, 3))
  return board
