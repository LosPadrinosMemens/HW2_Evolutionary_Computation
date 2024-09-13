from aux_functions import *

#######################
##     Functions     ##
#######################
symbols = sp.symbols('x1 x2')
x1, x2 = sp.symbols('x1 x2')
f_A = 100*(x1**2 -x2)**2 + (1-x1)**2

def rastrigin_function(n):
    """
    Returns a rastrigin sympy function given a determined number of variables

    Parameters:
    - n (int): number of variables in the rastrigin function

    Returns:
    - f(x) (sympy exp): the corresponding rastrigin function
    """
    x = sp.symbols(f'x1:{n+1}')
    
    # Generalized Rastrigin function
    f_x = 10 * n + sum(xi**2 - 10 * sp.cos(2 * sp.pi * xi) for xi in x)
    
    return f_x

f_B = rastrigin_function(2)
f_C = rastrigin_function(5)

########################
##   Initialization   ##
########################
def initialize(n, binary, precision_digits = 4, constraints = None):
    """
    Returns a random initialization for a problem given the number of variables, whether its binary encoding or real, and the
    constraints of the problem.

    Parameters:
    - n (int): number of variables in the rastrigin function.
    - binary (Boolean): True if binary encoding, False if real encoding.
    - precision_digits (int): number or digits of precision if the representation is binary.
    - constraints (list of tuples): Defining lower and upper limits for each variable.
    
    Returns:
    x_init (np.ndarray): values for initialization of the problem.
    """
    if constraints is None: # Creating generic constraints
        constraints = [(-10, 10)] * n 
    else:
        if len(constraints) is not n:
            raise ValueError("Length of constraints must be the same as the number of variables") 
    if binary:
        x_init = []
        for low, high in constraints:
            # Step 1: Calculate the size of the binary representation
            size = int(math.log2((high - low) * 10**precision_digits) + 0.99)

            # Step 2: Generate a random value within the range (low, high)
            real_value = np.random.uniform(low, high)
            print(real_value)

            # Step 3: Discretize the real value (shifted and scaled)
            scaled_value = int((real_value - low) * 10**precision_digits)
            print(scaled_value)

            # Step 4: Convert the scaled value to binary string of required length
            binary_rep = format(scaled_value, f'0{size}b')

            x_init.append(binary_rep)

        return np.array(x_init)

    else:
        x_init = np.array([np.random.uniform(low, high) for low, high in constraints])
        return x_init

#######################
##     Selection     ##     
#######################
def roulete_wheel(population, f_x, binary, constraints = None, precision_digits = 4, minimization = True):
    """
    Implementation of the Roulette wheel selection

    Parameters:
    - population (list of np.ndarray): list of individuals
    - f_x (sympy exp): the fitness function
    - binary (Boolean): True if binary encoding, False if real encoding
    - constraints (list of tuples): Defining lower and upper limits for each variable.
    - precision_digits (int): number or digits of precision if the representation is binary.
    - minimization (Boolean): whether the f_x is a minimization problem or not.

    Returns:
    - x_sel (np.ndarray): Selected individual
    """
    population_fitness = np.zeros(len(population))

    if binary:
        population = decode_population(population, constraints, precision_digits)

    for i, ind in enumerate(population):
        population_fitness[i] = eval_sympy(f_x, ind)

    # Correcting for negative values
    if np.min(population_fitness) < 0:
        population_fitness = population_fitness+np.abs(np.min(population_fitness))

    # Correcting for minimization problems
    if minimization:
        population_fitness = np.max(population_fitness) - population_fitness
    
    print(population_fitness)
    pop_prob_cum = np.cumsum(population_fitness / np.sum(population_fitness))
    print(pop_prob_cum)
    target = np.random.uniform(0, 1)
    print(target)
    sel_index = binary_search(pop_prob_cum, target)

    return population[sel_index]

def tournament_selection(population, f_x, binary, constraints = None, precision_digits = 4, minimization = True, q = 2, p = 0):
    """
    Implementation of the Roulette wheel selection

    Parameters:
    - population (list of np.ndarray): list of individuals
    - f_x (sympy exp): the fitness function
    - binary (Boolean): True if binary encoding, False if real encoding
    - constraints (list of tuples): Defining lower and upper limits for each variable.
    - precision_digits (int): number or digits of precision if the representation is binary.
    - minimization (Boolean): whether the f_x is a minimization problem or not.
    - q (int): number of individuals that participate in the tournament.
    - p (float): probability of flipping the decision (for probabilistic approach 0.5 < p <= 1, for deterministic p = 0)

    Returns:
    - x_sel (np.ndarray): Selected individual
    """
    population_fitness = np.zeros(len(population))

    if binary:
        population = decode_population(population, constraints, precision_digits)

    for i, ind in enumerate(population):
        population_fitness[i] = eval_sympy(f_x, ind)

    # Correcting for minimization problems
    if minimization:
        population_fitness = np.max(population_fitness) - population_fitness
    
    print(population_fitness)
    shuffled_indices = np.random.permutation(len(population))
    print(shuffled_indices)
    winners = []

    for i in range(0, len(shuffled_indices), q):
        group_indices = shuffled_indices[i:i+q] # Tournament indices
        group_values = population_fitness[group_indices] # Tournament fitness values
        if p == 0:    
            winners.append(group_indices[np.argmax(group_values)]) # Selecting winner
        else:
            sorted_group_indices = group_indices[np.argsort(-group_values)]
            probabilities = [(1 - p) ** i * p for i in range(len(sorted_group_indices))]
            probabilities[-1] = 1 - sum(probabilities[:-1])
            winner_index = np.random.choice(sorted_group_indices, p=probabilities)
            winners.append(winner_index)

    return np.array(winners)