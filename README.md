# Continuing Environments for Reinforcement Learning (`csuite`)


CSuite is a collection of carefully-curated synthetic environments designed for
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

```python
import csuite

env = csuite.load('catch')
observation = env.start()
agent = create_agent(...)
action = agent.start(observation)
for _ in range(NUM_STEPS):
  observation, reward = env.step(action)
  action = agent.step(reward, observation)
```

### Using `csuite` with dm_env interface

For a codebase that uses the [`dm_env`](https://github.com/deepmind/dm_env) interface, use the `DMEnvFromCSuite` wrapper class:

```python
import csuite

env = csuite.load('catch')
dm_env_env = csuite.dm_env_wrapper.DMEnvFromCSuite(env)

timestep = env.reset()
agent = create_agent(...)
action = agent.reset(timestep)
for _ in range(NUM_STEPS):
  timestep = env.step(action)
  action = agent.step(timestep)
```

