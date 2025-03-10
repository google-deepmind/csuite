# PuckWorld (continuous, 1D)
| Overview          | Specification               |
|-------------------|-----------------------------|
| Observation Space | 3-d array (specified below) |
| Action Space      | 1-d array (specified below) |
| Reward Space      | [-5, 0]                     |
| Loading String    | `'puckworld_continuous_1d'` |

## Description

> This version only has movement in the x-direction. 
> The agent can choose an action in [-1, 1] units only along the x axis. The observation space is also smaller, with information
only about the positions and velocity along the x-axis (see below).
> Everything else is same as 
> [the continuous-action version](https://github.com/abhisheknaik96/csuite/blob/main/docs/environments/puckworld_continuous.md)
> of the environment.
>
> This 1D version serves as a good diagonstic problem.

The aim for an agent is to reach a randomly appearing goal
in the environment.
The goal location changes after every fixed number of time steps.
The reward obtained by the agent per step is
inversely proportional to the distance of the agent from the goal,
and is maximized by going to the goal location as soon as possible.

This environment is a simpler variant of [PLE](https://github.com/ntasfi/PyGame-Learning-Environment)'s
[PuckWorld game](https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/puckworld.py).

## Observations

The observation is a 3-dimensional array of floats:
1. puck position along the x-axis
2. puck velocity along the x-axis
3. goal position along the x-axis

The x positions are within [0, `W`], the y positions are fixed to `H/2`, the x velocity is within [-`max_velocity`, `max_velocity`], and the y velocity is zero.
The origin (0, 0) is at the top-left corner.
The default values are `W`=`H`=5.
Given the dynamics (see the next section),
`max_velocity` turns out to be `a`/`k`.
The goal changes position every `G` steps, which defaults to 300.

## Actions

There is one continuous action along the x-axis: [-1, 1] units.

The actions cause an acceleration `action*a` in the corresponding direction.
Friction parameter `k` causes the velocity to decay in both directions.
As a result, the velocity along an axis changes according to:
`v_dot = A - k v`,
where `dt` is the size of simulation step and `A` denotes the acceleration along that axis (could be -`a`, 0, or `a`).
The default values are `a`=0.1, `k`=0.5, `dt`=0.05.

## Rewards

The reward signal is the negative distance of the puck
from the goal. So the reward is zero at the goal location and negative otherwise.
