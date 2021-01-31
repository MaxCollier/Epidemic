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

Problem B provides some complexity... Through brute force, and defining n as count of non-immune cells within a world, the search time complexity is O(2^n).

### An Iterative Solution...

A quicker solution than brute force is to use a randomized iterative threshold approach. To solve, we would decide on some threshold 't' where we will say "that within 't' attempts of randomized world states that infects 'i' vunerable people of aim to infect the entire population, that if we do not find a solution, we can assume i + 1 sick people will provide the best optimization possible, assuming that 't' is large enough". Here, we would start with a high 'i' value relative to 'n' and we would decrement 'i' each time we find a solution within 't' attempts. 

There are two issues I have with this solution...

The first, what would be an appropriate threshold for 't'? One way is that we could guess upon a resonable threshold, that is "I would think 1,000,000 is a large number of attempts to try find a solution, so I think that should be good". Another way is to instead provide a threshold of time rather than attempts. We might say "if the algorithm cannot find a solution within 10 minutes, then we assume the best possible optimization is the one previously found". A final means would be to use statistics. 

For example, we could look at an inital world state with n=100, we could look and s  , assuming that all of our random guesses made are unique.

I guess that 

The theory of large numbers applies. 

look back at previous attempts


This leads to me to my second point, if there is an extremely low probability of randoming finding an optimal solution, why don't we try another solution? 

we define a time range and match a compute a threshold and estimate a threashold. 
or we can estimate an effective threashold. 
how do we know how effective  

First, how do we know what to assign as an appropriate threshold for 't'. One way to solve this is with statistics. In a world where n=1000 and i=100, out of the possible 2^100 solutions that exist, only 10 initial world states would satisfy the world. This creates a success chance of 7.89e-30 per random state to win. 
We could say 10,000 or maybe 100,000 or more. 
The benifit with this is that we don't incur the risk of path traversals. 

One issue is that as 'i' gets smaller, the amount of potential working states reduces, making it very unlikely or impossible 

Secondly, there is always the possiblity at any threashold that the chance of the optimal solution not being found exists. Statisically, given high 'i' - lots of possible solutions to be found - and high 'n'. 

this in reduced Instead of solving the second mode through means of randomized iterative  TODO find actual name... the algorithm uses an optimal solve approach through means quicker than brute force.

Creates every permutation with two people? (check this) 
Then add


