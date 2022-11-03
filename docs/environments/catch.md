# Catch
| Overview          | Specification                             |
|-------------------|-------------------------------------------|
| Observation Space | Array of shape (10, 5) with binary values |
| Action Space      | \{0, 1, 2\}                               |
| Reward Space      | \{-1, 0, 1\}                              |
| Loading String    | `'catch'`                                 |

## Description
Catch as many falling objects as possible by controlling a breakout-like paddle
positioned at the bottom of a 10 x 5 board.

In the episodic version of Catch, an episode terminates after a single ball
reaches the bottom of the screen. This is a continuing version of Catch, in
which a new ball appears at the top of the screen with 10% probability, in a
column chosen uniformly at random. This means that multiple balls can be present
on the screen, but only one new ball can be *added* at each timestep.

At each timestep, balls present on the screen will fall by one pixel. Balls only
move downwards on the column they are in. The paddle can either stay in place,
or move by one pixel to the left or right at each timestep. Balls successfully
caught by the paddle give a reward of +1, and balls that fail to be caught by
the paddle give a reward of -1.

## Observations
The observation is an array of shape `(10, 5)`, with binary values:
zero if a space is empty; 1 if it contains the paddle or a ball. The initial
observation has one ball present at the top of the screen, in a column
chosen uniformly at random.

## Actions
There are three discrete actions.
* 0: Move the paddle one pixel to the left.
* 1: Keep the paddle in place.
* 2: Move the paddle one pixel to the right.

## Rewards
* If the paddle catches a ball, a reward of +1 is received.
* If the paddle fails to catch a ball, a reward of -1 is received.
* Otherwise the reward is 0.
