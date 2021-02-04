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

# test compute diagonals to see what it's used for then write some docs

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
    For each grid position, the indices of each neighbouring grid position is found and represented by an array 'indices'.
    The retrieval of an neighbours grid positions can be found using:

    >>> indices[i][j]

    where i is the row position and j in the column position.

    Parameters
    ----------
    shape : tuple
        dimensions of world

    Returns
    -------
    indices : list<list<np.ndarray>>
        index retrieval for neighbouring grid positions

    """
    (x_dim, y_dim) = shape
    
    masks = np.zeros((*shape, *shape), dtype=np.bool)

    for i in range(x_dim):
        for j in range(y_dim): 

            indices_mask = masks[i, j]

            _n = (i-1, j)
            _e = (i, j+1)
            _s = (i+1, j)
            _w = (i, j-1)

            if i != 0:
                indices_mask[_n] = True
            if j != y_dim-1:
                indices_mask[_e] = True
            if i != x_dim-1:
                indices_mask[_s] = True
            if j != 0:
                indices_mask[_w] = True
            
    indices = [[np.where(masks[i, j]) for j in range(y_dim)] for i in range(x_dim)]

    return indices


def calculate_neighbour_masks(shape):
    """
    Finds an array for the list of neighbours positions for each position in the universe. This is done by looping
    through each position, finding the indices of each neighbouring position and adding them if the neighbouring 
    positions are legitimate.

    Parameters
    ----------
    shape : tuple
        dimensions of universe

    Returns
    -------
    indices : list<list<np.array>>
        cache list of neighbours positions for each position in universe 

    """

    (x_dim, y_dim) = shape
    
    masks = np.zeros((*shape, *shape), dtype=np.bool)

    for i in range(x_dim):
        for j in range(y_dim): 

            indices_mask = masks[i, j]

            _n = (i-1, j)
            _e = (i, j+1)
            _s = (i+1, j)
            _w = (i, j-1)

            if i != 0:
                indices_mask[_n] = True
            if j != y_dim-1:
                indices_mask[_e] = True
            if i != x_dim-1:
                indices_mask[_s] = True
            if j != 0:
                indices_mask[_w] = True

    return masks


def get_sick_indices(shape):
    """
    Finds an array for the list of neighbours positions for each position in the universe. This is done by looping
    through each position, finding the indices of each neighbouring position and adding them if the neighbouring 
    positions are legitimate.

    Parameters
    ----------
    shape : tuple
        dimensions of universe

    Returns
    -------
    indices : list<list<np.array>>
        cache list of neighbours positions for each position in universe 

    """
    (x_dim, y_dim) = shape
    
    indices = np.zeros((*shape, *shape), dtype=np.bool)

    for i in range(x_dim):
        for j in range(y_dim): 

            indices_mask = indices[i, j]

            _nn = (i-2, j)
            _ee = (i, j+2)
            _ss = (i+2, j)
            _ww = (i, j-2)
            _nw = (i-1, j-1)
            _ne = (i-1, j+1)
            _se = (i+1, j+1)
            _sw = (i+1, j-1)

            if i >= 2:
                indices_mask[_nn] = True
            if j < y_dim-2:
                indices_mask[_ee] = True
            if i < x_dim-2:
                indices_mask[_ss] = True
            if j >= 2:
                indices_mask[_ww] = True
            if i > 0 and j > 0:
                indices_mask[_nw] = True
            if i > 0 and j < y_dim -1:
                indices_mask[_ne] = True
            if i < x_dim -1 and j < y_dim -1:
                indices_mask[_se] = True
            if i < x_dim -1 and j > 0:
                indices_mask[_sw] = True

    return indices


def spread_virus(state, indices):
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
    (dim_x, dim_y) = state.shape

    while has_updated:
        has_updated = False

        for i in range(dim_x):
            for j in range(dim_y):

                person = state[i, j]
                if person != VULNERABLE:
                    continue

                find_nieghbours = indices[i][j]
                neighbours = state[find_nieghbours]
                
                if np.sum(neighbours == SICK) >= 2:
                    next_state[i][j] = SICK
                    has_updated = True

        state = next_state

    return state


def solve_with_identity(state, neighbour_indices):
    """
    Finds optimal solution for 'find_minimum_sick' when there are no sick cases.
    """
    
    shape = state.shape
    (x_dim, y_dim) = shape
    (min_dim, max_dim) = np.sort(shape)
    
    world = np.zeros(shape)
    identity = np.identity(min_dim)

    if x_dim == y_dim:
        world = identity
    else:
        for i in range(0, min_dim + max_dim, min_dim + 1):
            if x_dim > y_dim:
                world[i:i+min_dim, :min_dim] = identity[:max_dim-i, :]
            else:
                world[:min_dim, i:i+min_dim] = identity[:, :max_dim-i]

        if not np.any(world[:, -1]) or not np.any(world[-1, :]):
            world[-1, -1] = SICK

    return world


def compute_starting_diagonals(state):
    """
    Initial means of creating a search array
    """

    # helper for flipping identity across veritical axis
    flip = lambda _ident: np.flip(_ident, axis=1)
    
    # cached variables
    shape = state.shape
    (x_dim, y_dim) = shape
    (min_dim, max_dim) = np.sort(shape)

    # logic below finds all diagonals
    dim_diff = max_dim - min_dim
    identities = { dim_num: np.identity(min_dim - dim_num, dtype=np.bool) for dim_num in range(min_dim) }
    _identities = { dim_num: flip(np.identity(min_dim - dim_num, dtype=np.bool)) for dim_num in range(min_dim) }

    # make all states and then just apply mask for where there is less that 1 case
    dim_diff = max_dim - min_dim
    minimal_states = np.zeros((0, *shape))
    x_diff = dim_diff if x_dim > y_dim else 0
    y_diff = dim_diff if x_dim < y_dim else 0

    # finds dialogals that are full length (min_dim)
    for idx in range(max_dim - min_dim + 1):
        x_idx = idx if x_dim > y_dim else 0
        y_idx = idx if x_dim < y_dim else 0
        x_offset = idx + min_dim if x_dim > y_dim else min_dim
        y_offset = idx + min_dim if x_dim < y_dim else min_dim

        # creates array copies
        left = state.copy()
        right = state.copy()

        # fills in where dialogals should go
        left[x_idx:x_offset, y_idx:y_offset][identities[0]] = SICK
        right[x_idx:x_offset, y_idx:y_offset][_identities[0]] = SICK

        # merges states into a single matrix
        array_stack = np.stack((left, right))
        minimal_states = np.concatenate((minimal_states, array_stack))

    # finds diagonals that aren't full length (min_dim)
    for idx in range(1, min_dim):

        # creates array copies
        top_left = state.copy()
        top_right = state.copy()
        bottom_left = state.copy()
        bottom_right = state.copy()

        # fills in where dialogals should go
        top_left[:-idx - x_diff, :-idx - y_diff][_identities[idx]] = SICK
        top_right[:-idx - x_diff, idx + y_diff:][identities[idx]] = SICK
        bottom_left[idx + x_diff:, :-idx - y_diff][identities[idx]] = SICK
        bottom_right[idx + x_diff:, idx + y_diff:][_identities[idx]] = SICK

        # merges states into a single matrix
        array_stack = np.stack((top_left, top_right, bottom_left, bottom_right))
        minimal_states = np.concatenate((minimal_states, array_stack))

    return minimal_states
     

def find_minimum_sick(state, nieghbour_indices):
    """
    Solves for solution to mode b using an evolutionary performance alogrithm to find the minimum number of additional sick
    people required to infect the entire population.

    Parameters
    ----------
    state : np.array
        initial world state

    nieghbour_indices: list<list<np.ndarray>>
        index retrieval for neighbouring grid positions. see 'calculate_neighbours' for explaination

    Returns
    -------
    state : np.array
        a world state with the least number of sick people to infect the whole population, including people who were
        initially sick

    """
    
    if np.sum(state == SICK) != 0: 
        state = spread_virus(state, nieghbour_indices)

    if np.sum(state == VULNERABLE) == 0: 
        return state

    if np.sum(state == VULNERABLE) == state.size:
        return solve_with_identity(state, nieghbour_indices)

    shape = state.shape
    final_state = state.copy()
    final_state[state == VULNERABLE] = SICK
    indices = get_sick_indices(shape)
    neighbours = calculate_neighbour_masks(shape)

    # create initial search states
    minimal_states = np.zeros((np.sum(state == VULNERABLE), *shape))
    minimal_states[:] = state
    for idx, (i, j) in enumerate(zip(*np.where(state == VULNERABLE))):
        minimal_states[idx, i, j] = SICK

    # plays world out before hand infecting as many people as possible
    world_states = minimal_states.copy()
    for i in range(world_states.shape[0]):
        world_states[i] = spread_virus(minimal_states[i], nieghbour_indices)

    # creates hash value keys for each state, useful for finding duplicates
    world_hashes = np.array([make_hash(world_state) for world_state in world_states])

    # loop is used to find best state, THIS IS WHERE THE MAGIC HAPPENS
    generation = 0
    best_minimal = np.ones(shape)
    best_minimal[state == IMMUNE] = -1.
    while world_states.shape[0] > 0:
        
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
        #   - minial: the positions of each sick person that aim to infect the whole board
        #   - world: the worlds states after expanding the infection with 'spread_virus'
        #   - hash: keys that reprsent the state of each board.
        #
        _minimal_queue = np.zeros((*world_states.shape, *shape))
        _world_queue = np.zeros((*world_states.shape, *shape))
        _hash_queue = np.empty(world_states.shape, dtype=world_hashes.dtype)
 
        # mask to identify which items in the queue were updated
        _util_mask = np.zeros(world_states.shape, dtype=np.bool)

        # finds how placing a sick person in each part of the grid infects the rest of the grid. Process cannot be computed
        # via a parrellel as non pure python context cannot be supplied to processes.
        for (p, i, j) in zip(*np.where(world_states == VULNERABLE)):

            # checks if adding a sick person at i, j will infect other parts of the board
            if np.any(world_states[p][indices[i, j]] == SICK):

                # copies previous state
                _prev = world_states[p]
                _state = _prev.copy()
                _minimal = minimal_states[p].copy()

                # adds sick people 
                _state[i, j] = SICK
                _minimal[i, j] = SICK

                # finds the new state of the board
                _util_mask[p, i, j] = True
                _minimal_queue[p, i, j] = _minimal
                _world_queue[p, i, j] = _state = spread_virus(_state, nieghbour_indices)
                _hash_queue[p, i, j] = make_hash(_state)
        
        #
        # finds states that didn't update in previous generation and check if their current states can be progressed
        # 

        # creates empty array structures
        new_world_states = np.zeros((0, *shape))
        new_minimal_states = np.zeros((0, *shape))
        new_hashes = np.empty(0, dtype=world_hashes.dtype)

        # looks at blocked states and checks if other areas of the world can a sick person be placed
        uncompleted_mask = np.sum(_util_mask == False, axis=(1,2)) == 0
        for (idx,) in zip(*np.where(uncompleted_mask)):
            
            # selects values from array
            new_state_created = False
            _world_state = world_states[idx]
            _minimal_state = world_states[idx]

            # looks at positions where sick people have not been placed
            for (i, j) in zip(*np.where(_world_state == VULNERABLE)):

                # checks that position does not directly neighbour a sick person, otherwise it can be considered a final state
                # position. 
                if np.sum(_world_state[neighbours] == SICK) == 0:
                    new_state_created = True
                    _new_world_state = _world_state.copy()
                    _new_minimal_state = _minimal_state.copy()

                    _new_world_state[i, j] = SICK
                    _new_minimal_state[i, j] = SICK
                    _new_world_state = spread_virus(_new_world_state, nieghbour_indices)
                    
                    new_hashes = np.append(new_hashes, make_hash(_new_world_state))
                    new_world_states = np.concatenate((new_world_states, _new_world_state.reshape((1, *shape))))
                    new_minimal_states = np.concatenate((new_minimal_states, _new_minimal_state.reshape((1, *shape))))

            # no future states to be generated so current state can be considered final and checked to see if it is the best
            if not new_state_created:
                for (i, j) in zip(*np.where(_world_state == VULNERABLE)):
                    _minimal_state[i, j] = SICK

                if np.sum(_minimal_state == SICK) < np.sum(best_minimal == SICK):
                    best_minimal = _minimal_state

        #
        # performs sorting of metrics and states for being able to remove duplicates effeciently.
        #

        # reduces arrays into states that where updated
        hashes = _hash_queue[_util_mask]
        world_states = _world_queue[_util_mask]
        minimal_states = _minimal_queue[_util_mask]

        # short curcuits solution to avoid errors below
        if world_states.size == 0:
            return best_minimal

        # when duplicate world states are found, we want to take the case that has the minimal people sick to infect to world state
        # https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
        _idx_sort = np.argsort(hashes)
        world_hashes, idx_start = np.unique(hashes[_idx_sort], return_index=True)

        # gathers indicies with fewest sick in minial state for duplicate and non-duplicate world states.
        _temp_sort = []
        _minimal_sorted = np.sum(minimal_states == SICK, axis=(1,2))[_idx_sort]
        _minimal_split = np.split(_idx_sort, idx_start[1:])
        for split in _minimal_split:
            idx = np.argmin(_minimal_sorted[split])
            _temp_sort.append(split[idx])

        _idx_sort = np.array(_temp_sort)
        
        # applies mask to remove duplicates world states
        world_states = world_states[_idx_sort]
        minimal_states = minimal_states[_idx_sort]

        # appends new states that have been generated to minimal_states
        world_hashes = np.append(world_hashes, new_hashes)
        world_states = np.concatenate((world_states, new_world_states))
        minimal_states = np.concatenate((minimal_states, new_minimal_states))

        # updates new best minimum mask, this requires one of the world states to have no vulnerable
        completed_mask = np.all(world_states == final_state, axis=(1, 2))
        if np.any(completed_mask):
            [_minimal_state] = minimal_states[completed_mask]
            if np.sum(_minimal_state == SICK) < np.sum(best_minimal == SICK):
                best_minimal = _minimal_state

        # removes cases where there are more states cannot be better than 'best_minimal'
        _count = np.sum(world_states == SICK, axis=(1,2))
        _minimal = np.sum(minimal_states == SICK, axis=(1,2))
        mask = ~completed_mask & (_minimal < np.sum(best_minimal == SICK))
        
        # applies filter to get the best 750 searches based on the ratio between how many people have become sick
        # from how many people sick there are
        if generation >= 2 and np.sum(mask) >= 500:
            
            # computes ratios for each state and find bins for how much each ratio will remove
            ratio = _count / _minimal
            _ratio = ratio[mask]
            ratios = np.unique(_ratio)

            # finds a ratio cut off which allows for at most 1000 search cases to persist
            remove_bins = np.array([ np.sum(_r < _ratio) for _r in ratios ])
            bin_idx = np.argmin(np.abs(remove_bins - 500))
            ratio_cutoff = ratios[bin_idx]

            # applies cutoff to mask
            mask[mask] &= ratio_cutoff < _ratio

        # removes states that have failed some condition stated in the few paragraphs above
        world_hashes = world_hashes[mask]
        world_states = world_states[mask]
        minimal_states = minimal_states[mask]

        generation += 1

        # uncomment to see how fast it is generating it is searching through solutions
        # print(minimal_states.shape)
        
    return best_minimal
    

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
        '--pool-maximum', '-n', type=int, default=500,
        help="Hyperparameter for algorithm defining the maximum number of best performing state instances that can exist "\
            "between generations. Only useful when '-b' is used."
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help="Show information between generations when evolutionary algorithm is running"
    )

    return parser.parse_args()


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
