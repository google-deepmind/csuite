# Continuing Environments for Reinforcement Learning (`csuite`)


CSuite is a collection of carefully-curated synthetic environments for
research in the continuing setting: the agent-environment interaction goes on
forever without limit, with no natural episode boundaries.

## Installation

Clone the source code into a local directory and install using pip:

```sh
git clone https://github.com/deepmind/csuite.git /path/to/local/csuite/
pip install /path/to/local/csuite/
```

`csuite` is not yet available from PyPI.

## Environment Interface

CSuite environments adhere to the Python interface defined in `csuite/environment/base.py`.
Find the interface [documentation here](https://rl-csuite.readthedocs.io/en/latest/api.html).

```python
import csuite

env = csuite.load("catch")
action_spec = env.action_spec()
observation = env.start()
print("First observation:\n", observation)

total_reward = 0
for _ in range(100):
  observation, reward = env.step(action_spec.generate_value())
  total_reward += reward

print("Total reward:", total_reward)
```

### Using `csuite` with dm_env interface

For a codebase that uses the [`dm_env`](https://github.com/deepmind/dm_env) interface, use the `DMEnvFromCSuite` wrapper class:

```python
import csuite

env = csuite.dm_env_wrapper.DMEnvFromCSuite(csuite.load("catch"))
action_spec = env.action_spec()

timestep = env.reset()
print("First observation:\n", timestep.observation)

total_reward = 0
for _ in range(100):
  timestep = env.step(action_spec.generate_value())
  total_reward += timestep.reward

print("Total reward:", total_reward)
```

