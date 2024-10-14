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
    - n (int): number of genes in the individuals.
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
        x_real = []
        x_binrep = []
        for low, high in constraints:
            # Step 1: Calculate the size of the binary representation
            size = int(math.log2((high - low) * 10**precision_digits) + 0.99)

            # Step 2: Generate a random value within the range (low, high)
            real_value = np.random.uniform(low, high)
            x_real.append(real_value)

            binary_rep = encode_binary(real_value,(low,high))

            x_binrep.append(binary_rep)

        return np.array(x_binrep)
    else:
        x_real = [np.random.uniform(low, high) for low, high in constraints]
        return np.array(x_real)

#######################
##     Crossover     ##
#######################
def point_crossover(parent1, parent2, n=1):
    """
    Given two parents np.arrays of strings, it returns the resulting child after n-point crossover for binary encoding.

    Parameters:
    - n (int): number of crossover points. Default is single point crossover
    - parent1 & parent2 (np.array): string of binary encoded value

    Returns:
    - child1 & child2 (np.array): resulting string of binary encoded values
    """
    parent1_length = np.vectorize(len)(parent1)
    parent2_length = np.vectorize(len)(parent2)

    parent1_indices = np.cumsum([0] + list(parent1_length))
    parent2_indices = np.cumsum([0] + list(parent2_length))

    parent1 = ''.join(parent1)
    parent2 = ''.join(parent2)

    N = min(len(parent1), len(parent2))
    if n > N:
        raise ValueError("Number of crossover points cannot be greater than length of parents")
    
    crossover_points = random.sample(range(1, N+1), n)
    #print(crossover_points)
    child1, child2 = "", ""
    prev_point = 0
    
    for i, point in enumerate(crossover_points + [N]): # <100101, 1010, 101>
        if i % 2 == 0:
            child1 += parent1[prev_point:point]
            child2 += parent2[prev_point:point]
        else:
            child1 += parent2[prev_point:point]
            child2 += parent1[prev_point:point]
        
        prev_point = point
    
    child1 = np.array([child1[parent1_indices[i]:parent1_indices[i+1]] for i in range(len(parent1_length))])
    child2 = np.array([child2[parent2_indices[i]:parent2_indices[i+1]] for i in range(len(parent2_length))])

    return child1, child2

def sbx(parent1, parent2, u=None, nc=2):
    """
    Given two parents arrays of real values, it performs simulated binary crossover returning the two children

    Parameters:
    - parent1 & parent2 (np.ndarray): arrays of real values
    - nc (int): n_c value, n=0 uniform distribution, 2<n<5 matches closely the simulation for single-point crossover

    Returns:
    - child1 & child2 (np.ndarray): resulting arrays of real values
    """
    b = spread_factor(u, nc)
    child1 = 0.5 * ((parent1 + parent2) - b * (parent2 - parent1)) 
    child2 = 0.5 * ((parent1 + parent2) + b * (parent2 - parent1))

    return child1, child2


def binomial_crossover_and_selection(parent, trial, n_genes, obj_func, pc=0.8):
    J = set_J(n_genes, pc)

    offspring = np.copy(parent)

    # Step 3: Perform the crossover
    for j in J:
        offspring[j] = trial[j]  # Replace with trial solution values at crossover points

    offspring_fitness = eval_sympy(obj_func, offspring)
    parent_fitness = eval_sympy(obj_func, parent)

    if offspring_fitness < parent_fitness:
        return offspring
    else:
        return parent

########################
##      Mutation      ##
########################
def binary_mutation(parent, n = 1, p = 1.0):
    """
    Given a parent string, returns the mutated child (bit flip mutation). Only works for binary encoded values

    Parameters:
    - parent (str): string of binary encoded value
    - n (int): number of mutations
    - p (float): probability of mutation. Default always mutate.

    Returns:
    - child (str): resulting mutated child.
    """
    N = len(parent) # Amount of bits
    if n > N:
        raise ValueError("Number of mutation points cannot be greater than length of parent")
    
    mutation_points = random.sample(range(0, N), n)
    mutation_mask = [random.random() <= p for _ in range(n)]
    child = list(parent)
    
    for i, point in enumerate(mutation_points):
        if mutation_mask[i]:
            child[point] = '1' if parent[point] == '0' else '0'
    
    return ''.join(child)

def parameter_based_mutation(y, constraints, t):
    """
    Given a real encoded parent, it returns a child that resulted from mutating the parent

    Parameters:
    - y (np.array): Real encoded values of parent (single parent)
    - constraints (list of tuples): Defining lower and upper limits for the real variable.
    - t (int): generation number

    Returns:
    - y_mut (np.array): Real encoded values of child (mutated)
    """
    if y.ndim != 1:
        raise ValueError("y should be a one dimension np.array")
    
    y_mut = np.empty_like(y)

    eta_m = 100 + t 
    for i, y_i in enumerate(y): # Independently mutate every value from y
        y_l, y_u = constraints[i]
        
        delta_max = y_u - y_l
        delta = min(y_i-y_l,(y_u-y_i))/(delta_max)

        beta_q = beta_q_factor(delta=delta, eta_m=eta_m)

        y_mut[i] = y_i + beta_q * delta_max

    return y_mut

def polynomial_mutation(X, F):
    """
    Perform polynomial mutation on a 2D point (x, y coordinates), using indices of X for lower and upper values.
    
    Args:
    X: numpy array with two values representing x and y coordinates.
    F: scaling factor for mutation (usually between 0.5 and 1).
    
    Returns:
    Mutated point U (with mutated x and y coordinates).
    """
    # Randomly sample the indices for X_lower and X_upper
    lower_index = np.random.randint(0, len(X))  # Random index for X_lower
    upper_index = np.random.randint(0, len(X))  # Random index for X_upper
    
    # Extract the values of X_lower and X_upper based on the sampled indices
    X_lower = X[lower_index]
    X_upper = X[upper_index]
    
    # Apply mutation (operation for x = [y, z], i.e., both coordinates)
    U = X + F * (X_upper - X_lower)
    
    return U

#######################
##     Selection     ##
#######################
def roulete_wheel(population, f_x, binary=True, constraints = None, precision_digits = 4, minimization = True):
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
    population_fitness = eval_population(population, f_x, binary, constraints, precision_digits)

    # Correcting for negative values
    if np.min(population_fitness) < 0:
        population_fitness = population_fitness+np.abs(np.min(population_fitness))

    # Correcting for minimization problems
    if minimization:
        population_fitness = np.max(population_fitness) - population_fitness
    
    #print(population_fitness)
    pop_prob_cum = np.cumsum(population_fitness / np.sum(population_fitness))
    #print(pop_prob_cum)
    target = np.random.uniform(0, 1)
    #print(target)
    sel_index = binary_search(pop_prob_cum, target)

    return population[sel_index]

def tournament_selection(population, f_x, binary=False, constraints = None, precision_digits = 4, minimization = True, q = 2, p = 0):
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
    
    #print(population_fitness)
    shuffled_indices = np.random.permutation(len(population))
    #print(shuffled_indices)
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

