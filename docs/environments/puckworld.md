# Puckworld
| Overview          | Specification                                             |
|-------------------|-----------------------------------------------------------|
| Observation Space | 6-d array of real numbers  |
| Action Space      | \{0, 1, 2, 3\}                                               |
| Reward Space      | between around -7 and 0                                   |
| Loading String    | `'puckworld'`                                              |


## Description

Move a puck towards a goal location which changes repeatedly.

The puck can be pushed in the four cardinal directions. 
Every push results in an instantaneous acceleration of $a=0.1$ units. 
The velocity of the puck changes in each direction as: 
$v_{t+1} \doteq v_t + (a_t - \mu v_t) \Delta t$, $x_{t+1} \doteq x_t + v_{t+1} \Delta t$, 
where $\mu=0.1$ is the coefficient of friction. 
These dynamics result in an upper bound of the velocity to be $|v_{\max}| = a/\mu$. 
The puck is in an arena of size `width`*`height` units, which are set to 5 and 5 by default. 
A collision against the wall results in the loss of 50% speed in the corresponding direction of motion.
The goal position is randomly reinitialized within the arena after every $\tau=300$ time steps.


## Observations

The observation is a 6-dimensional real array: 

0. x-coordinate of puck position in [0, `width`]
1. y-coordinate of puck position in [0, `height`]
2. x-coordinate of puck velocity in [$-|v_{\max}|$, $|v_{\max}|$]
3. y-coordinate of puck velocity in [$-|v_{\max}|$, $|v_{\max}|$]
4. x-coordinate of goal position in [0, `width`]
5. y-coordinate of goal position in [0, `height`]


## Actions

There are four discrete actions:
* 0: 0.1 units of acceleration up
* 1: 0.1 units of acceleration left
* 2: 0.1 units of acceleration down
* 3: 0.1 units of acceleration right


## Rewards
The reward is the negative distance of the puck from the goal. 
The minimum or maximum reward is when the puck is over the goal (zero reward) or furthest away ($5*\sqrt{2}$).
