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

"""Install script for setuptools."""

import os
from setuptools import find_namespace_packages
from setuptools import setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name='csuite',
    version='0.1.0',
    url='https://github.com/deepmind/csuite',
    license='Apache 2.0',
    author='DeepMind',
    description=(
        'A collection of continuing environments for reinforcement learning.'),
    long_description=open(os.path.join(_CURRENT_DIR, 'README.md')).read(),
    long_description_content_type='text/markdown',
    author_email='csuite@google.com',
    keywords='reinforcement-learning environment suite python machine learning',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=[
        'dm_env>=1.5',
        'gym>=0.19.0',
        'numpy>=1.18.0',
        'Pillow>=9.0.1',
        'absl-py>=0.7.1',
        'pytest>=6.2.5',
    ],
    zip_safe=False,  # Required for full installation.
    python_requires='>=3.7',
    classifiers=[
        # TODO(b/241264065): list classifiers.
    ],
)
