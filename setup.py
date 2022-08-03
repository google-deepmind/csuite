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


def _parse_requirements(path):

  with open(os.path.join(_CURRENT_DIR, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


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
    # TODO(zaheersm): Use a more friendly email than buganizer component alias.
    author_email='buganizer-system+1073031@google.com',
    keywords='reinforcement-learning environment suite python machine learning',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements.txt')),
    tests_require=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements-test.txt')),
    zip_safe=False,  # Required for full installation.
    python_requires='>=3.7',
    classifiers=[
        # TODO(b/241264065): list classifiers.
    ],
)
