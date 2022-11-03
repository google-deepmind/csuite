# Environment Interface
All CSuite environments adhere to the Python interface defined in the abstract
base class `csuite.Environment`. This base class specifies the standard methods
to implement in a CSuite environment, which are outlined below.

```{eval-rst}
.. autoclass:: csuite.Environment
```

## Loading a CSuite Environment
Environments in CSuite are specified by an identifying string and can be
initialized using the `load` function.

```python
import csuite

env = csuite.load('catch')
```

The list of available environments and their associated loading strings is
given in the `csuite.EnvName` class.

```{eval-rst}
.. autoclass:: csuite.EnvName
```

## API Methods
### Start
After initialization, `start` must be called to set the environment state. Since
all environments are continuing, `start` should only be called *once*, at the
beginning of the agent-environment interaction.

```{eval-rst}
.. automethod:: csuite.Environment.start
```

### Step
After `start` is called, `step` updates the environment by one timestep
given the action taken. The resulting observation and reward are returned.

```{eval-rst}
.. automethod:: csuite.Environment.step
```

### Render
All CSuite environments are expected to return an object serving to render
the environment for visualization at the current timestep.

```{eval-rst}
.. automethod:: csuite.Environment.render
```

### Get and Set State
The `get_state` and `set_state` methods permit environment state retrieval
and manipulation. These methods should only be used for reproducibility or
checkpointing purposes; thus, it is expected that these methods can sufficiently
manipulate the internal state to provide full reproducibility of the environment
dynamics (supplying the internal random number generator if applicable,
for example).

```{eval-rst}
.. automethod:: csuite.Environment.get_state

.. automethod:: csuite.Environment.set_state
```

### Observation and Action Specs
Environments are expected to return the specifications of the observation and
action space by calling `observation_spec` and `action_spec` respectively.
These methods should return structures of dm_env
[`Array` specs](https://github.com/deepmind/dm_env/blob/master/dm_env/specs.py)
which adhere exactly to the format of observations and actions.

```{eval-rst}
.. automethod:: csuite.Environment.observation_spec

.. automethod:: csuite.Environment.action_spec
```
