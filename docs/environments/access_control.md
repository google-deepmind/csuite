# Access-Control
|     Overview      |    Specification   |
|-------------------|--------------------|
| Observation Space | \{0, 43\}          |
| Action Space      | \{0, 1\}           |
| Reward Space      | \{1, 2, 4, 8\}     |
| Loading String    | `'access_control'` |


## Description
Given access to a set of 10 servers and an infinite queue of customers with
different priorities, decide whether to accept or reject the next customer
in line based on their priority and the number of free servers. 

Each customer has a uniformly random priority of 1, 2, 4, or 8. At each time
step, the customer at the head of the queue is either accepted (assigned to one
of the servers) or rejected (removed from the queue, with a reward of zero).
If accepted, customers provide the agent with a reward equal to their priority.
In either case, on the next time step the next customer in the queue is
considered. Customers cannot be accepted when all servers are busy, and
busy servers are freed with probability 0.06 at each step.

This problem is based on the Access-Control queueing task in *Reinforcement
Learning* by Sutton and Barto (see [Example 10.2, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf#page=274)).

## Observations
The state space is represented as tuples `(num_busy_servers, incoming_priority)`
and an observation is a single integer encoding the current state. With there
being 0 to 10 busy servers and 4 possible customer priorities, there is a total
of 44 discrete states.

## Actions
There are two discrete actions.
* 0: Reject the incoming customer.
* 1: Accept the incoming customer.

## Rewards
The incoming customer has a priority equal to 1, 2, 4, or 8.
* If the customer is accepted, the reward equals the customer's priority.
* Otherwise the customer is rejected and the reward equals 0.


