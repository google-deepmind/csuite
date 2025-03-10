# PuckWorld (continuous)
| Overview          | Specification               |
|-------------------|-----------------------------|
| Observation Space | 6-d array (specified below) |
| Action Space      | 2-d array (specified below) |
| Reward Space      | [-7, 0]                     |
| Loading String    | `'puckworld_continuous'`    |

## Description

> This version has continuous actions. 
> The agent can choose an action in [-1, 1] units along the x and y axes.
> Everything else is same as 
> [the discrete-action version](https://github.com/abhisheknaik96/csuite/blob/main/docs/environments/puckworld.md)
> of the environment.

The aim for an agent is to reach a randomly appearing goal
in the environment.
The goal location changes after every fixed number of time steps.
The reward obtained by the agent per step is
inversely proportional to the distance of the agent from the goal,
and is maximized by going to the goal location as soon as possible.

This environment is a simpler variant of [PLE](https://github.com/ntasfi/PyGame-Learning-Environment)'s
[PuckWorld game](https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/puckworld.py).

## Observations

The observation is a 6-dimensional array of floats:
1. puck position along the x-axis
2. puck position along the y-axis
3. puck velocity along the x-axis
4. puck velocity along the y-axis
5. goal position along the x-axis
6. goal position along the y-axis

The x positions are within [0, `W`], y positions within [0, `H`], and
both x and y velocities within [-`max_velocity`, `max_velocity`].
The origin (0, 0) is at the top-left corner.
The default values are `W`=`H`=5.
Given the dynamics (see the next section),
`max_velocity` turns out to be `a`/`k`.
The goal changes position every `G` steps, which defaults to 300.

## Actions

There are two continuous actions:
* Action 1: [-1, 1] along the x-axis
* Action 2: [-1, 1] along the y-axis

The actions cause an acceleration `a` in the corresponding direction.
Friction parameter `k` causes the velocity to decay in both directions.
As a result, the velocity along an axis changes according to:
`v_dot = A - k v`,
where `dt` is the size of simulation step and `A` denotes the acceleration along that axis (could be -`a`, 0, or `a`).
The default values are `a`=0.1, `k`=0.5, `dt`=0.05.

## Rewards

The reward signal is the negative distance of the puck
from the goal. So the reward is zero at the goal location and negative otherwise.
