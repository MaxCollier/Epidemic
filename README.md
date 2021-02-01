# Epidemic

A near-optimal algorithm solution for solving the based game Epidemic. 

## Game and Rules

In worlds where a virus is spreading, people have gathered in groups to protect themselves. For a person to get sick, they must be stading standing either beside, infront or behind two other sick people, unless that person is immune. Each person at any time can be either:
* S -> Sick
* . -> Vulnerable
* I -> Immune

There are two ways to play the game:
* **A**: Find the final world state once the virus has spread through the population given an initial world state. _(easy)_
* **B**: Given a world state where there are no sick people, find an initial world state that will be able to infect an entire vulnerable population with as few sick people as possible. _(difficult)_

## Run

Make sure to install _python3.6+_ and _numpy_ into your environment. You can the program `$ python3 epidemic.py { -a | -b } < input.txt`. Input is read through stdin and must be run in mode _a_ or _b_. Initial world states can be of any shape. Look to _examples.txt_ for examples.

### Explaining the algorithm for problem B

To find an initial world where the smallest number of sick are able to infect the entire population, we have choices for which algorithm we could implement. The simplest one, brute force, is attempting every possible world state and checking for which one is the best. This method has a drawback that it does not compute well with large problems. The number of possible world states scales to 2^n where _n_ is all the number of non-immune people that exist within the population. This means that brute force expodential time to solve, making the algorithm extremely slow the higher _n_ grows.

#### An Iterative Solution...

A quicker solution than brute force is to use a randomized iterative threshold approach. To solve, we would decide on some threshold _t_ where we will say that "within _t_ attempts, we would expect that at least one randomized initial state with _i_ infectous people will infect the entire non-immune population. Otherwise, we can assume that _i+1_ infectous people will provide the best solution possible, **assuming that _t_ is large enough**". Here, we would start with a high _i_ value relative to _n_ and we would decrement _i_ each time we find a solution within _t_ attempts.

The first question we need to ask is what would be an appropriate threshold for _t_? One way is to guess what a good threshold would be. That is to say, "I would think _x_ number of attempts would be a large enough to try find a solution ". This can work, but it is wishful and nieve of any context of problem.

Another way is to search a percentage of the search space, and conclude that it will be unlikely to find a solution if we search any further. This technique would be better as we are choosing a threshold in context of size of the problem and we can provide some inference from the result. As an example, given a world where _n_ = 100, we might decide to search 0.0001% of the possible world states. This means that we would search `1.27e+24 ≅ (0.0001 * 2^100) / 100` world states. If we were to search and not find any solutions, we could infer using the inversion of the [law of truly large numbers](https://en.wikipedia.org/wiki/Law_of_truly_large_numbers), that it can will be unlikely that a solution would exist at a rate more than 1 in 1,000,000 (working is `ans = 100 / 0.0001`, as 0.0001 represents a percentage). 

Another way would be to instead provide a threshold of time rather than attempts. We might say "if the algorithm cannot find a solution within _'48' hours_, then we assume the best possible optimization is the one previously found". This can work best in cases where it is difficult to know how our attempts theshold will take and we have time to let the algorithm run.

Another question we need to ask is how confident can I be using this alogrithm? We might have infered upon the upper bounds for the rate of successful solutions in the search space, but can we be sure there cannot be any? An issue with using this technique is that, when attempting to find solutions when 'n' is high, we are unlikely to find a result, even when running the algorithm for long periods of time. To illustrate, say if _n_=100 and _i_=20, and with the possible world shape and considering the arrangement of immune people there were a small number of 1 trillion solutions. This would provide just 7.88e-19% of the all the possible solutions - try runing `1 - ((2**100 - 1e+12) / 2**100)` in your interpter, you might need to add zeros to _1e+12_ to get some accurate precision! It can be expected that the most extreme optimizations when _i_ is at it's lowest possible value, the number of possible solutions will tend towards zero (but not equal). Hence, using such an algorithm relies on (an emormous amount of) luck for finding an the optimal solution.

It should pointed out that choosing such an algorithm neglects the possibility that an optimal or near-optimal deterministic solution can exist. This has lead me to my algorithm...

### Pooled Greedy

todo...
