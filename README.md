# Epidemic

An evolutionary algorithm for solving the based game Epidemic. 

## Game and Rules

In worlds where a virus is spreading, people have gathered in groups to protect themselves. For a person to get sick, they must be stading either beside, infront or behind two other sick people, unless that person is immune. Each person at any time can be either:
* S -> Sick
* . -> Vulnerable
* I -> Immune

There are two ways to play the game:
* **A**: Find the final world state once the virus has spread through the population given an initial world state. _(easy)_
* **B**: Given a world state where there are no sick people, find an initial world state that will be able to infect an entire vulnerable population with as few sick people as possible. _(difficult)_

## Run

Make sure to install _python3.6+_ and _numpy_ into your environment. You can the program `$ python3 epidemic.py { -a | -b } < input.txt`. Input is read through stdin and must be run in mode _a_ or _b_. Initial world states can be of any shape. Look to _examples\_a.txt_ or _examples\_b.txt_ for examples.

You can also run `$ python3 epidemic.py --help` for more information and other options.

### Finding an Algorithm to Solve B

For finding a solution for mode B, we have different algorithms we can implement. The simplest, brute force, attempts to check every possible world state to search for which one is the best. Such a solution can be useful when we can grauntee a small number of states to search, however, this does not perform well against larger problems. The number of possible world states scales to 2^n where _n_ is all the number of non-immune people that exist within the population. This expodential increase means brute force will stuggle to perform for problems with reasonable sizes of _n_.

A quicker solution than brute force is to use a randomized iterative threshold approach. To solve, we would decide on some threshold _t_ where we will say that "within _t_ attempts, we would expect that at least one randomized state with _i_ infectous people will infect the entire population. Otherwise, we can assume that _i+1_ infectous people will provide the best solution possible, **assuming that _t_ is large enough**". Here, we would start with a high _i_ value relative to _n_ and we would decrement _i_ each time we find a solution within _t_ attempts. This solution would work better than brute force, however, it is unlikely to provide the best solution as described below[below](#improving-the-iterative-algorithm).

## A Middle Ground...

An algorithm that would work as a middle ground between brute force and randomness would be to use an evolutionary algorithm. This would work by searching down the tree of all possible states one level at a time - starting from an initial state. Once gathering all the branching states on some level, we then reduce our collection to the best performing set of states under some branching threshold _l_. We can compare states by checking the performance of how well each state has spread the virus. Once the number of branches has been reduced within the threshold, we then repeat, finding the next set of possible states from our current branches until we find a solution.

This algorithm can perform significantly better than the brute force and the randomized iterative threshold approach as it provides a tradeoff between time efficiency and correctness. If we want to find relatively correct answer fast, we can set a low branching limit to _l_. If we want to find a solution that with a high expectation for correctness, we can set a high branching limit to _l_. The benefit of this will be that we will avoid searching over solutions which are unlikely to provide an optimal solution from states that do not perform well from previous placements of infectious people. 

The drawback to this algorithm is that it operates in a greedy fashion, only ever looking one state ahead. However, due to the nature of the problem being combinatorial, it can be understood that states that do not perform well at the start and states that do perform well at the start can converge to the same solution. This is because branching state can place infectious people down in different orders before reaching their shared final solution. Hence, a state that it not performing well that may lead to the global minima can be represented through a state that may be performing well, **assuming that _l_ is large enough**.

By searching multiple branches at once, we infact reduce the bias that the algorithm will only find local minina solutions rather than finding a global optimal solution as we are searching a greater represantion of the state tree. This greater representation means that we are more likely to uncover more of the error, or 'performance' surface that will lead some branching solutions to the optimal.

#### Improving The Iterative Algorithm

The first issue we have when implementing an randomized iterative algorithm is how do we know when to stop if we can't find a solution? Using some threshold _t_ can be a useful means of making this decision. However, then we will ask the question of what is an appropriate threshold for _t_?

One way would be to guess what a good threshold would be. That is to say, "I would think _x_ number of attempts would be a large enough to try find a solution, and any more attempts would be unlikely to find a solution". This can work, but it is wishful and nieve of any context and scale of the problem.

Another way is to search a percentage of the search space, and conclude that it will be unlikely to find a solution if we search any further. This technique would be better as we are choosing a threshold in context of size of the problem and we can provide some inference from the result. As an example, given a world where _n_ = 100, we might decide to search 0.0001% of the possible world states. This means that we would search `1.27e+24 ≅ (0.0001 * 2^100) / 100` world states. If we were to search and not find any solutions, we could infer using the inversion of the [law of truly large numbers](https://en.wikipedia.org/wiki/Law_of_truly_large_numbers), that it can will be unlikely that a solution would exist at a rate more than 1 in 1,000,000 (working is `ans = (1 / 0.0001) * (1 / 100)`, as 0.0001 represents a percentage). 

Another way would be to instead provide a threshold of time rather than attempts. We might say "if the algorithm cannot find a solution within _'48' hours_, then we assume the best possible optimization is the one previously found". This can work best in cases where it is difficult to know how our attempts theshold will take and we have time to let the algorithm run.

Another question we need to ask is how confident can I be using this alogrithm? We might have infered upon the upper bounds for the rate of successful solutions in the search space, but can we be sure there cannot be any? An issue with using this technique is that, when attempting to find solutions when 'n' is high, we are unlikely to find a result, even when running the algorithm for long periods of time. To illustrate, say if _n_=100 and _i_=20, and with the possible world shape and considering the arrangement of immune people there were a small number of one trillion solutions. This would provide just 7.88e-19% of the all the possible solutions - try runing `1 - ((2**100 - 1e+12) / 2**100)` in your interpter, you might need to add zeros to _1e+12_ to get an accurate precision from your percentage! It can be expected that the most extreme optimizations when _i_ is at lowest amount possible, the number of possible solutions will be less than one trillion, as the number of possible solutions will tend towards zero (but not equal) with decrements of _i_. Hence, using such an algorithm relies on (an emormous amount of) luck for finding an optimal solution. 

