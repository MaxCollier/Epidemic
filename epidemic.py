import numpy as np
import math
import sys

SICK_KEY = 'S'
SICK = 1

IMMUNE_KEY = 'I'
IMMUNE = -1

VULNERABLE_KEY = '.'
VULNERABLE = 0

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


def make_hash(state):
    """
    Hashes a given world state.
    """

    key = ""
    (x_dim, y_dim) = state.shape
    for i in range(x_dim):
        for j in range(y_dim):
            key += state_decoder[state[i, j]]
    
    return key


def create_task(state, has_sick):
    """
    Creates a 'task' based on the state of the state. Tasks are to be used by the 'main' function for running the selected
    algorithm (mode a or b). 

    Parameters
    ----------
    state : np.array
        current world state

    state : np.array
        current world state

    Returns
    -------
    object :    
        state : np.array
            initial world state
        mask : np.array
            identifies sick and immune existing individuals
        has_sick : bool
            whether any cell contains a sick person

    """

    row_lens = np.array([ len(row) for row in state ])

    # unifromly shaped world state, no map manipulation required
    if np.all(row_lens == row_lens[0]):
        return {
            'state': np.array(state),
            'has_sick': has_sick,
            'world_mask': np.ones(np.array(state).shape, dtype=np.bool)
        }

    # TODO check if this works correctly.. No if statement so it might blank out 
    world_mask = []
    new_state = []
    len_target = np.max(row_lens)
    for i, (row, row_len) in enumerate(zip(state, row_lens)):
        row_mask = [ True for _ in range(row_len) ]
        for j in range(row_len, len_target):
            row.append(IMMUNE)
            row_mask.append(False)
        new_state.append(row)
        world_mask.append(row_mask)

    return {
        'has_sick': has_sick,
        'state': np.array(new_state),
        'world_mask': np.array(world_mask, dtype=np.bool),
    }
    

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
    universe = []
    has_sick = False
    while lines or universe:
        
        # handles case where final new line doesn't exist from file
        if lines:
            line = lines.pop(0)
        elif universe:
            line = ''

        # empty line represents end of universe  
        if not line:

            # no task exists if no universe exists
            if not universe:
                continue
            
            # appends task
            tasks.append(create_task(universe, has_sick))

            # accounting
            universe = []
            has_sick = False
            continue

        # checks if a sick case exists in universe
        if SICK_KEY in line:
            has_sick = True

        # adds row to universe
        row = [state_encoder[token] for token in line]
        universe.append(row)

    return tasks


def calculate_grid_neighbours(shape):
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
            
    indices = [[np.where(masks[i, j]) for j in range(y_dim)] for i in range(x_dim)]

    return indices


def get_neighbour_indicies(shape):
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


def find_final_state(state):
    """
    Finds the final state of a universe after each infected case has been infected. This is done by looping through each
    position and updating vulnerable cases to sick when possible until there are no more cases to become sick. The
    condition for making a vulnerable person sick is that at least two neighbours need to be sick as well.

    Parameters
    ----------
    state : np.array
        initial state of universe

    Returns
    -------
    state : np.array
        final state of universe

    """

    has_updated = True
    next_state = state.copy()
    (dim_x, dim_y) = state.shape
    indices = calculate_grid_neighbours(state.shape)

    while has_updated:
        has_updated = False

        for i in range(dim_x):
            for j in range(dim_y):

                person = state[i, j]
                if person != VULNERABLE:
                    continue

                mask = indices[i][j]
                neighbours = state[mask]
                
                if np.sum(neighbours == SICK) >= 2:
                    next_state[i][j] = SICK
                    has_updated = True

        state = next_state

    return state


def short_curcuit_compute(state):
    """
    Finds optimal solution for 'find_minimum_sick' when there are no sick cases.
    """
    
    # cached variables
    shape = state.shape
    (x_dim, y_dim) = shape
    (min_dim, max_dim) = np.sort(shape)
    
    mask = np.zeros(shape, dtype=np.bool)
    mask[:min_dim, :min_dim] = np.identity(min_dim, dtype=np.bool)
    
    for i in range(min_dim + 1, max_dim, 2):
        if x_dim == min_dim:
            mask[0, i] = True 
        else:
            mask[i, 0] = True 

    # handles exception case where final row/column won't get filled in
    x_sum = np.sum(mask, axis=0)
    y_sum = np.sum(mask, axis=1)
    if x_sum[-1] == 0 or y_sum[-1] == 0:
        mask[-1, -1] = True

    # asserts whole table will get sick
    table = np.zeros(shape)
    table[mask] = SICK
    result = find_final_state(table)
    assert np.all(result == SICK)

    return table


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
    diagonals = []
    diag_info = []
    axis_x = x_dim >= y_dim
    dim_diff = max_dim - min_dim
    identities = { dim_num: np.identity(min_dim - dim_num, dtype=np.bool) for dim_num in range(min_dim) }
    _identities = { dim_num: flip(np.identity(min_dim - dim_num, dtype=np.bool)) for dim_num in range(min_dim) }

    # state = np.arange(35).reshape(7,5)
    # make all states and then just apply mask for where there is less that 1 case
    state_index = 0
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
     

def find_minimum_sick(state):
    """
    Finds the state where the minimum amount of sick people exist to fill the whole universe. This is done by searching 
    through creating a search tree and representing the state in numpy arrays. Branching paths are broken up by divisors
    which create the path for how leave nodes are grouped.

    Parameters
    ----------
    state : np.array
        initial state of universe, has no sick people

    Returns
    -------
    state : np.array
        universe with the minimum amount of sick people to infect the whole population

    """

    assert np.sum(state == SICK) == 0, "No sick cases can exist in universe where we are trying to find the minimum sick cases"
    
    if np.sum(state == VULNERABLE) == 0:
        return state
    
    # cached variables
    shape = state.shape
    (x_dim, y_dim) = shape
    (min_dim, max_dim) = np.sort(shape)
    indices = get_sick_indices(shape)
    neighbours = get_neighbour_indicies(shape)

    # places immune cases back to their positions
    immune_mask = state == IMMUNE
    num_immune = np.sum(immune_mask)

    # HANDLES - no immune cases on grid (short curcuit compute)
    if np.sum(immune_mask) == 0:
        return short_curcuit_compute(state)

    # create initial search states
    minimal_states = np.zeros((np.sum(~immune_mask), *shape))
    minimal_states[:, immune_mask] = IMMUNE
    for idx, (i, j) in enumerate(zip(*np.where(~immune_mask))):
        minimal_states[idx, i, j] = SICK

    # finds final state array for comsparisons
    final_state = state.copy()
    final_state[~immune_mask] = SICK
    
    # removes duplicate board states, can occur when there are many immune cases
    minimal_states = np.unique(minimal_states, axis=0)
    world_states = minimal_states.copy()

    # plays world out before hand infecting as many people as possible
    for i in range(world_states.shape[0]):
        world_states[i] = find_final_state(minimal_states[i])

    # creates hash value keys for each state, useful for finding duplicates
    world_hashes = np.array([make_hash(world_state) for world_state in world_states])

    # loop is used to find best state, THIS IS WHERE THE MAGIC HAPPENS
    generation = 0
    best_minimal = np.ones(shape)
    best_minimal[immune_mask] = -1.
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
        #   - world: the worlds states after expanding the infection with 'find_final_state'
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
                _world_queue[p, i, j] = _state = find_final_state(_state)
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
                    _new_world_state = find_final_state(_new_world_state)
                    
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
            mask_len = np.sum(mask)

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


def format_and_print(universe, mode, world_mask):
    """
    Encodes map back to original encoding from the nomial encoding given during 'read_input'.

    Parameters
    ----------
    universe : np.array
        final state of the universe

    has_sick : bool
        flag for if true, print the amount of sick cases that exist in universe

    """

    if mode == 'b':
        num_sick = np.sum(universe == SICK)
        print(num_sick)

    str_universe = ""
    for i, row in enumerate(universe):
        for j, cell in enumerate(row):
            if world_mask[i, j]:
                str_universe += state_decoder[cell]
            else:
                str_universe += " "
        str_universe += '\n'
    print(str_universe)



def main():
    """
    Reads input for universes and solves them for either completing the state of the universe by observing a sickness
    spread across the population or find the minimum number of cases for a universe that will make the entire
    population sick.
    """

    assert len(sys.argv) == 2, "missing mode flag (-a or -b)"
    
    # read command line flags
    task_handler = None
    if sys.argv[1] == '-a':
        mode = 'a'
        task_handler = find_final_state
    elif sys.argv[1] == '-b':
        mode = 'b'
        task_handler = find_minimum_sick
    else:
        raise ValueError("flag needs to be either -a or -b")

    tasks = read_input()

    for task in tasks:

        universe = task['state']
        has_sick = task['has_sick']
        world_mask = task['world_mask']

        # task b cannot have any sick people according to etude spec
        if mode == 'b' and has_sick:
            raise ValueError("Task b cannot contain sick people")

        result = task_handler(universe)

        format_and_print(result, mode, world_mask)


if __name__ == "__main__":
    main()
