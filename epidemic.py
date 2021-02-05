import numpy as np
import argparse
import math
import sys

SICK = 1
SICK_KEY = 'S'

IMMUNE = -1
IMMUNE_KEY = 'I'

VULNERABLE = 0
VULNERABLE_KEY = '.'

BRANCHING_LIMIT = 20

state_encoder = {
    SICK_KEY: SICK,
    IMMUNE_KEY: IMMUNE,
    VULNERABLE_KEY: VULNERABLE,
}

state_decoder = {
    SICK: SICK_KEY,
    IMMUNE: IMMUNE_KEY,
    VULNERABLE: VULNERABLE_KEY,
}

# todo, this can be improved with only checking states where there is alraedy ONE sick person
# beside a vulnerable person...

# does the initial search states need to exist if we have a variable to count how many sick their are?

# read up about greedy algorithms and see if mine is just an alternative to something else
# that exists.


def make_hash(world):
    """
    Hashes a given world world state.
    """

    (x_dim, y_dim) = world.shape
    key = "".join([state_decoder[world[i, j]] for i in range(x_dim) for j in range(y_dim)])
    
    return key


class Task:
    """
    Tasks merely provides an facade abstraction to finding extra information on the initial world state. These states are
    additionally encoded from list of lists to numpy ndarrays. These arrays are much faster than python arrays as they 
    leverage 'C' implementation and provide powerful mapping and multi-indexing capabilities.

    Attributes
    ----------
    world : np.ndarray
        initial state of the world
    init_sick : int
        initial number of sick people in the world
    world_mask : np.ndarray
        mask to find actual area of 'world'. useful for worlds that provide in non-uniform shapes

    """

    def __init__(self, init_state):
        """
        Initializes the fields of task. There are two methods for initializing the world state encoding, which are numpy
        array functions and indice loops. The first method can be used when the initial world shape is uniform (i.e when
        all rows are of equal length) and the second method can be used when for non-uniform worlds. 

        """
        row_lens = np.array([ len(row) for row in init_state ])
        uniform_dims = np.all(row_lens == row_lens[0])
        max_dim = np.max(row_lens)
    
        if uniform_dims:
            self.world = np.array(init_state)
            self.init_sick = np.sum(self.world == SICK)
            self.world_mask = np.ones(self.world.shape, dtype=np.bool)

        else:
            shape = (row_lens.size, max_dim)
            padded_world_state = np.zeros(shape)
            rm_padded_world_mask = np.zeros(shape, dtype=np.bool)
            padded_world_state[:] = IMMUNE

            for i, row in enumerate(init_state):
                for j, person in enumerate(row):
                    padded_world_state[i, j] = person
                    rm_padded_world_mask[i, j] = True

            self.world = padded_world_state
            self.world_mask = rm_padded_world_mask
            self.init_sick = np.sum(self.world == SICK)

        self.neighbour_indices = calculate_neighbours(self.world.shape)


def calculate_neighbours(shape):
    """
    Returns an array which provides masks to find neighbouring cells for each grid position. The retrieval of an neighbours
    grid positions can be found using:

    >>> neighbour_masks[i, j]

    where i is the row position and j in the column position of the cell we want to find the neighbours of. This will return
    a 2D mask which then be applied to retrieve the values nieghbouring cells with:

    >>> world_state[neighbour_masks[i, j]]

    Investigating the structure of 'neighbour_masks', we can notice four dimensions to the array. The first should be 
    considered the query parameters. Say if "I want to find the neighbours on cell (2, 3) in the world", then applying:
    
    >>> world_state[neighbour_masks[2, 3]]

    Will retrieve the neighbouring cells. These next two dimensions then represent the dimensions of the world_state which
    will hold boolean values representing whether a cell is a neighbour of the queried (i, j). 
    
    To learn more on the topic of masks see:
    https://numpy.org/doc/1.20/user/basics.indexing.html#boolean-or-mask-index-arrays

    Parameters
    ----------
    shape : tuple
        dimensions of world

    Returns
    -------
    neighbour_masks : np.ndarray
        masks for finding neighbouring cells for each grid position

    """

    (x_dim, y_dim) = shape
    neighbour_masks = np.zeros((*shape, *shape), dtype=np.bool)

    # loops over all positions of grid
    for (i, j) in zip(*np.where(np.ones(shape, dtype=np.bool))):

            # selects the neighbour mask to work with by referencing a section the underlying array
            neighbour_mask = neighbour_masks[i, j]

            # calculates NESW neighbour positions 
            _n = (i-1, j)
            _e = (i, j+1)
            _s = (i+1, j)
            _w = (i, j-1)

            # checks if nieghbour exists before updating
            if i != 0:
                neighbour_mask[_n] = True
            if j != y_dim-1:
                neighbour_mask[_e] = True
            if i != x_dim-1:
                neighbour_mask[_s] = True
            if j != 0:
                neighbour_mask[_w] = True

    return neighbour_masks


def spread_virus(state, neighbours):
    """
    Performs rule to spread the virus. If someone vulnerable is standing beside two sick people, then that person becomes
    sick themself. This rule is repeated until the disease is not able to spread any futher. This world state is then 
    returned. 

    Parameters
    ----------
    state : np.ndarray
        initial state of world

    indices : list<list<np.ndarray>>
        cached indice retrieval map for neighbouring grid positions. see 'calculate_neighbours' for more details

    Returns
    -------
    state : np.ndarray
        final state of world once virus has spread

    """

    has_updated = True
    next_state = state.copy()
    grid_positions = np.where(np.ones(state.shape, dtype=np.bool))

    # spreads virus while it is still able to spread
    while has_updated:
        has_updated = False

        # loops across every grid position
        for (i, j) in zip(*grid_positions):

            # if person is vulnerable and has two sick neighbours
            if state[i, j] == VULNERABLE and np.sum(state[neighbours[i, j]] == SICK) >= 2:
                has_updated = True
                next_state[i, j] = SICK

        # updates buffer
        state = next_state

    return state


def solve_with_identity(state):
    """
    Given a world where there is only vunlerable people, a solution can be found by mapping an identity matrix along the 
    longest dimension. Between each copy of the identity, a gap can be left which will be filled by either side (there 
    will be a sick person on either side once the virus spreads). If the gap placed on the last row/column of the world,
    then the person in the bottom corner will be made sick (otherwise the last row/column would only have sick people
    on one side). 

    Parameters
    ----------
    state : np.array
        initial world state

    Returns
    -------
    state : np.array
        a world state with the least number of sick people to infect the whole population

    """

    shape = state.shape
    (x_dim, y_dim) = shape
    (min_dim, max_dim) = np.sort(shape)
    
    world = np.zeros(shape)
    identity = np.identity(min_dim)

    # identity solution can be applied straight away
    if x_dim == y_dim:
        world = identity
    else:

        # applies the identity martix through the world with a gap between each identity
        for i in range(0, min_dim + max_dim, min_dim + 1):

            # figures out it it's applied along the row and column
            if x_dim > y_dim:
                world[i:i+min_dim, :min_dim] = identity[:max_dim-i, :]
            else:
                world[:min_dim, i:i+min_dim] = identity[:, :max_dim-i]

        # in the case the final row/column has no sick people, insert a sick person into the bottom left corner to solve 
        # problem
        if not np.any(world[:, -1]) or not np.any(world[-1, :]):
            world[-1, -1] = SICK

    return world
     

def find_minimum_sick(init_state, neighbours):
    """
    Solves for solution to mode b using an evolutionary performance alogrithm to find the minimum number of additional sick
    people required to infect the entire population.

    spr_branches represents all the nodes of the state tree local at their local decisions with forsight of a single move.
    The exist and move all at the same depth. Hence once one node returns a completed solution, then we will know that it
    is the best one the algorithm has made. 

    # queues are actaully pre-allocated arrays of memory that provide a means of quickly computing, selecting, searching
    # for world states given quering slices. the queues have five dimensions each, the first three for diveriging branches
    # from state p, where a new sick cell is placed at (i, j). the second two dimensions are for the state of the world.
    #
    # an analogy to think of how this works is you could compare these 'queues' to some database (data being packed together
    # in an effective encoding) and where a slice of three parameters (p: previous board state, i, j: new position of sick
    # person) will return the world state after a person has been placed there. This is much more effective than using pure
    # python array-likes as these arrays are effectively encoded in C structures where you can perfrom C implement metric 
    # queries such as sum, mean, mask operations and other things.
    # 
    # it's important to note that it's expensive to compute the next world state from a current state, hence a computiation
    # limiting condition has been added so that world states that will grauntee to infect neighbours will be 
    #
    # The different arrays are:
    #
    #   - minimal: the positions of each sick person that aim to infect the whole board
    #   - world: the worlds states after expanding the infection with 'spread_virus'
    #   - hash: keys that reprsent the state of each board.
    #

    # uncomment to see how fast it is generating it is searching through solutions
    # print(minimal_states.shape)

    Parameters
    ----------
    init_state : np.array
        initial world state

    neighbours: np.ndarray
        result of 'calculate_neighbours' function in respect of current world 

    Returns
    -------
    state : np.array
        a world state with the least number of sick people to infect the whole population, including people who were
        initially sick

    """

    # spreads virus from initial state if possible    
    if np.sum(init_state == SICK) == 0: 
        world_state = init_state
    else:
        world_state = spread_virus(init_state, neighbours)

    # already at a final state so return init_state 
    if np.sum(world_state == VULNERABLE) == 0: 
        return init_state

    # no sick or immune people in world, quick alternative solution
    if np.sum(init_state == VULNERABLE) == init_state.size:
        return solve_with_identity(init_state)

    # cached variables
    dims = init_state.shape
    hash_type = f"<U{world_state.size}"
    final_state = init_state.copy()
    final_state[init_state == VULNERABLE] = SICK

    # The initial branches need to be created in order for algorithm to work. Having at least one option to infect is 
    # guaranteed as virus has already been spread where 'world_state' is not a final state.
    vunerable_mask = world_state == VULNERABLE
    branching_states_minimal = np.zeros((np.sum(vunerable_mask), *dims)) # states before virus spreads
    branching_states = np.zeros((np.sum(vunerable_mask), *dims)) # states after virus spreads
    branching_hashes = np.zeros(np.sum(vunerable_mask), dtype=hash_type)

    # initializes all branches with original states
    branching_states_minimal[:] = init_state
    branching_states[:] = world_state

    # updates the decision tree by placing a sick person for each vulnerable position on each branch. This is to represent
    # the branching decisions that can be made
    for branch_idx, (i, j) in enumerate(zip(*np.where(vunerable_mask))):

        # places sick person in one position for each branch
        branching_states_minimal[branch_idx, i, j] = SICK

        # spread state after placing sick person for each branch
        branch_state = branching_states[branch_idx]
        branch_state[i, j] = SICK
        branching_states[branch_idx] = spread_virus(branch_state, neighbours)

        # creates hashes for each branching decsision
        branching_hashes[branch_idx] = make_hash(branching_states[branch_idx])

    # THIS IS WHERE THE MAGIC HAPPENS
    while True:

        # solution returned once found (no secondary criteria)
        solution_found = np.all(branching_states == final_state, axis=(1, 2))
        if np.any(solution_found):
            return branching_states_minimal[solution_found][0]

        # pre-allocated buffers from next branching states
        prealloc_shape = (*branching_states.shape, *dims)
        next_branches = np.zeros(prealloc_shape)
        next_branches_minimal = np.zeros(prealloc_shape)
        next_branches_mask = np.zeros(branching_states.shape, dtype=np.bool)
        next_branching_hashes = np.empty(branching_states.shape, dtype=hash_type)

        # makes every possible move from all current states in 'next_branches'
        for (p, i, j) in zip(*np.where(branching_states == VULNERABLE)):
            next_branches_mask[p, i, j] = True

            # copies previous state
            state = branching_states[p].copy()
            minimal = branching_states_minimal[p].copy()

            # inserts sick people 
            state[i, j] = SICK
            minimal[i, j] = SICK
            state = spread_virus(state, neighbours)

            # places next states into buffers 
            next_branches[p, i, j] = state
            next_branches_minimal[p, i, j] = minimal
            next_branching_hashes[p, i, j] = make_hash(state)

        # finds next branches from buffer by applying mask
        next_branches = next_branches[next_branches_mask]
        next_branches_minimal = next_branches_minimal[next_branches_mask]
        next_branching_hashes = next_branching_hashes[next_branches_mask]

        # reduces the size of branching state if too large
        if branching_states.shape[0] > BRANCHING_LIMIT:

            # buffers branching states that are selected on performance
            buffer_branches = np.zeros((0, *dims))
            buffer_branches_minimal = np.zeros((0, *dims))
            
            # performance is based off how many sick people exist
            branch_perfs = np.sum(next_branches == SICK, axis=(1,2))

            # unique performance values are discovered from highest to lowest
            unique_perfs = np.unique(branch_perfs)[::-1]

            # adds states until branching limit is reached
            buffer_count = 0
            for perf in unique_perfs:

                # finds all states at current performance
                branches_mask = perf == branch_perfs
                branches_count = np.sum(branches_mask)

                # adds branches at current performance directly to buffer if enough space
                if buffer_count + branches_count <= BRANCHING_LIMIT:

                    # updates buffer count
                    buffer_count += branches_count
                    
                    # extends buffers with next best performing branches
                    buffer_branches = np.concatenate((
                        buffer_branches,
                        next_branches[branches_mask]
                    ))
                    buffer_branches_minimal = np.concatenate((
                        buffer_branches_minimal,
                        next_branches_minimal[branches_mask]
                    ))

                else:
                    # fills remaining space in buffer with a randomized selection of branches at current performace  
                    limit = BRANCHING_LIMIT - buffer_count
                    randomize = np.random.permutation(branches_count)

                    # extends buffers with these randomized branches
                    buffer_branches = np.concatenate((
                        buffer_branches, 
                        next_branches[branches_mask][randomize][:limit]
                    ))
                    buffer_branches_minimal = np.concatenate((
                        buffer_branches_minimal,
                        next_branches_minimal[branches_mask][randomize][:limit]
                    ))

            # buffer now becomes next_branches as it holds the best selection of branches within limit
            next_branches = buffer_branches
            next_branches_minimal = buffer_branches_minimal
            next_branching_hashes = np.array([make_hash(next_branches[i]) for i in range(BRANCHING_LIMIT)], dtype=hash_type)

        # moves buffered 'next_branches' data into 'branching_states'
        branching_states = next_branches
        branching_hashes = next_branching_hashes
        branching_states_minimal = next_branches_minimal


def read_input():
    """
    Reads initial world states through stdin and encodes them into arrays. Additional information about the initial world
    state will be generated and bundled by the call to 'create_task'. Each task refers to each listed world state read from
    input, where each world will be processed seperately.

    Input is read line by line, forming and encoding the world state in the buffer 'state_buf'. When the end of a world is
    reached, it's task is created and added 'tasks' which is returned by the function once all world states are read from
    the input.

    Returns
    -------
        tasks : list<dict>
            containing world states with additional information. see 'create_task' for more...

    """

    lines = [line.strip() for line in sys.stdin.readlines()]

    tasks = []
    cur_world = []
    while lines or cur_world:
        
        line = lines.pop(0) if lines else ''

        if line:
            cur_world.append([state_encoder[token] for token in line])
        else:
            if len(cur_world) == 0:
                continue

            task = Task(cur_world)
            tasks.append(task)
            cur_world = []

    return tasks


def format_and_print(world, task, args):
    """
    Encodes map back to original encoding from the nomial encoding given during 'read_input'. Provides extra details if 
    the program was run in mode b.

    Parameters
    ----------
    world : np.ndarray
        final solution of world state for either mode a or b
    task : object
        holds world_mask to find mask of original world state. see 'Task' class for more details
    args : argparse.NameSpace
        holds state for whether the program was run in mode b

    """

    world_mask = task.world_mask
    final_sick = np.sum(world == SICK)
    addtional_sick = final_sick - task.init_sick

    if args.mode_b:
        if task.init_sick == 0:
            print(f"Minimum of {final_sick} sick people required to infect the entire population.")
        else:
            print(f"An additional {addtional_sick} sick people required from to infect the entire population, {final_sick} total.")

    str_universe = ""
    for i, row in enumerate(world):
        for j, cell in enumerate(row):
                str_universe += state_decoder[cell] if world_mask[i, j] else " "
        str_universe += '\n'
    print(str_universe)


def handle_args():
    """
    Command line arguments handler.

    Returns
    -------
    args : dict    
        key value pairs for command line arguments

    """

    global BRANCHING_LIMIT

    parser = argparse.ArgumentParser(
        description="Algorithms to solve the game Epidemic. Mode B provide a semi-evolutionary algorithm to solve problem"\
                   " in polynomial time."
    )
    modes = parser.add_mutually_exclusive_group(
        required=True
    )
    modes.add_argument(
        '-a', action='store_true', dest="mode_a",
        help="Given an inital world, find the final world state once the virus has spread."
    )
    modes.add_argument(       
        '-b', action='store_true', dest="mode_b",
        help="Given a world with no sick people, find an initial world state with the smallest number of sick people that"\
            "can infect the entire non-immune population"
    )
    parser.add_argument(
        '--pool-maximum', '-n', type=int, default=BRANCHING_LIMIT, dest="branching_limit",
        help="Hyperparameter for algorithm defining the maximum number of best performing state instances that can exist "\
            "between generations. Only useful when '-b' is used."
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help="Show information between generations when evolutionary algorithm is running"
    )

    # updates global variable
    args = parser.parse_args()
    BRANCHING_LIMIT = args.branching_limit

    return args


def main():
    """
    Reads input for universes and solves them for either completing the state of the universe by observing a sickness
    spread across the population or find the minimum number of cases for a universe that will make the entire
    population sick.

    """
    args = handle_args()
    if args.mode_a:
        task_handler = spread_virus
    else:
        task_handler = find_minimum_sick

    tasks = read_input()
    for task in tasks:
        result = task_handler(task.world, task.neighbour_indices)
        format_and_print(result, task, args)


if __name__ == "__main__":
    main()
