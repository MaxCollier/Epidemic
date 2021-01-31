# Epidemic

An algorithm for finding an optimal solution the rule based game Epidemic.

## Game and Rules

There are worlds where disease is spreading. In these worlds, people standing side by side together. Each person at any one time can be either:
    -   S -> Sick
    -   . -> Vulnerable
    -   I -> Immune

If two sick people are standing beside someone who is vunlnerable, then the vunlnerable person will become sick. This process can repeat until there is no more individuals who can get sick or it spreads throughout the entire population. 

## Run

You can the program `$ python3 epidemic.py { -a | -b } < input.txt`. The initial world states are to be read through stdin. The program must be run in one of two modes, 'a' or 'b'. This argument must be specified through the command line.

### Mode A:
Given a world state where there is some people are sick, find the final world state once the disease has spread.

### Mode B:
Given a world state where there no person is sick, find a initial world state that will be able to infect an entire population. 

## Input

The world can be of any dimensions and shape - it does not need to be uniform. Example worlds can be found in `examples.txt`.

## Algorithm Explained

Problem B provides some complexity... Through brute force, and defining n as count of non-immune cells within a world, the search time complexity is O(2^n), expodential time. 

### An Iterative Solution...

A quicker solution than brute force is to use a randomized iterative threshold approach. To solve, we would decide on some threshold 't' where we will say that "within 't' attempts of randomized world states that infects 'i' vunerable people of aim to infect the entire population, that if we do not find a solution, we can assume i + 1 sick people will provide the best optimization possible, assuming that 't' is large enough". Here, we would start with a high 'i' value relative to 'n' and we would decrement 'i' each time we find a solution within 't' attempts.

The first question we need to ask is what would be an appropriate threshold for 't'? One way is that we could guess upon a resonable threshold, that is to say, "I would think 10,000,000 would be a large number of attempts to try find a solution, so I think that should be good". This can work, but it is wishful and nieve to the full extend of the data.

Another way that provides some inference to our decision would be to choose a percentage of all the possible states we could search, and to then conclude it is unlikely that we will find a solution if we search any further. For example, using a world where n=100, we will find that 0.01% of the total possible permutations is (0.01 * 2^100) / 100 'approx' = 1.27e+26 world states. Now, we could choose this number to be our upper limit. 

Another way is to instead provide a threshold of time rather than attempts. We might say "if the algorithm cannot find a solution within 48 hours, then we assume the best possible optimization is the one previously found". This can work best in cases where it is hard to estimate how long searching to an attempt threashold can take.

An issue with using this technique is that for trying to find optimized solutions when 'n' is high, we are unlikely to find a result when running the algorithm for long periods of time. For example if 'n'=100 and 'i'=20, and with the possible world shape and arrangement of immune people provided a small number of 1 trillion solutions, this provides just 7.8e-18 of the all the possible solutions. It can be expected that the most extreme optimizations of i, the number of possible solutions will tend towards zero (but never equal). Hence, using such an algorithm relies on - a great deal of - luck for finding an optimal solution.

It should point out that choosing such an algorithm neglects the possibility that an optimal or near-optimal deterministic solution exists.

### A Path search
