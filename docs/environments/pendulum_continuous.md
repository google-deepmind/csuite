# Pendulum
| Overview          | Specification                                             |
|-------------------|-----------------------------------------------------------|
| Observation Space | 3-d array: [ $\cos\theta$, $\sin\theta$, $\dot{\theta}$ ] |
| Action Space      | [-1, 1] (see below)                                       |
| Reward Space      | between around -20 to 0                                   |
| Loading String    | `'pendulum'`                                              |


## Description

> This version has continuous actions. 
> The agent can choose a torque in [-1, 1] units.
> Everything else is same as 
> [the discrete-action version](https://github.com/abhisheknaik96/csuite/blob/main/docs/environments/pendulum.md)
> of the environment.

Starting from a hanging down position, swing up a single pendulum to the
inverted position and maintain the pendulum in this position.

Most of the default arguments model the pendulum described in "A Comparison of
Direct and Model-Based Reinforcement Learning" (Atkeson & Santamaria, 1997).
The key differences are:
1) This environment is now continuing, i.e. the "trial length" is infinite.
2) Instead of directly returning the angle of the pendulum in the observation,
this environment returns the cosine and sine of the angle.
3) There are only three discrete actions (apply a fixed magnitude of torque in
either direction or apply no torque) as opposed to a continuous torque value.
4) The reward signal is inversely proportional to the pendulum's angular
distance from the upright position and inversely proportional to its angular
velocity (more details below).

The pendulum's motion is described by the equation:
$\dot{\theta} = \tau - \mu \dot{\theta} - g \sin\theta$ ,
where $\theta$ is the angle with the vertical, $\mu$ is the friction coefficient,
$\tau$ is the torque, $g$ is the acceleration due to gravity,
$\dot{\theta}$ is the first derivative of theta w.r.t. time,
and $\ddot{\theta}$ is the second derivative.
The mass and length of the pendulum is set to values that result in the
above simplified equation of motion.

## Observations

The observation is a 3-dimensional array encoding the cosine and sine of the
pendulum's angle as well as its angular velocity: [ $\cos\theta$, $\sin\theta$, $\dot{\theta}$ ].
The pendulum starts at the bottom ( $\theta=0$ ) with zero velocity.
The first two elements of the observation are inherently bounded in [-1, 1];
the third element is bound in the code by the parameter `max_speed`, whose
default value is `np.inf`.

## Actions

There is one continuous action: [-1, 1] units of torque.
The sign indicates the direction of torque.

## Rewards
The default reward function is dense:
reward = - $(\pi - \theta)^2 - 0.1 * \dot{\theta}^2$

There is a sparse-reward variant available:
* +1 reward if the pendulum is within a small range of the upright position:
( $\pi$ - reward_angle, $\pi$ + reward_angle)
* 0 reward otherwise
