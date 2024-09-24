import random
import numpy as np
import sympy as sp
import math

def eval_sympy(obj_func, x):
    """
    Parameters:
    obj_func (sympy exp or list): Objective function or list of derivates
    x (np.ndarray):  Array of values to substitute into the objective function

    Returns:
    - f(x) (float): The evaluated function value
    """
    if isinstance(obj_func, list):
        if all(isinstance(el, list) for el in obj_func): # Hessian Matrix
            return np.array([[eval_sympy(func, x) for func in row] for row in obj_func])
        else:                                            # Jacobian Vector
            return np.array([eval_sympy(func, x) for func in obj_func])
    
    elif isinstance(obj_func, sp.Expr):
        sorted_symbols = sorted(obj_func.free_symbols, key=lambda s: s.name)
        subs_dict = {symbol: value for symbol, value in zip(sorted_symbols, x.tolist())}
        result = obj_func.subs(subs_dict)
    return float(result)

def eval_population(population, obj_func, binary, constraints, precision_digits=4):
    """
    Parameters:
    - population (list of np.ndarray): list of individuals
    - f_x (sympy exp): the fitness function
    - binary (Boolean): True if binary encoding, False if real encoding
    - constraints (list of tuples): Defining lower and upper limits for each variable.
    - precision_digits (int): number or digits of precision if the representation is binary.
    
    Returns:
    - population_fitness(list)
    """
    population_fitness = np.zeros(len(population))

    if binary:
        population = decode_population(population, constraints, precision_digits)

    for i, ind in enumerate(population):
        population_fitness[i] = eval_sympy(obj_func, ind)

    return np.array(population_fitness)

def constraint_checker(x_val, constraints, binary, precision_digits=4):
    """
    Checks if the new the solution of the problem is within the constraint of a problem

    Parameters:
    - x (np.ndarray): New solution to the problem
    - constraints (list of list): Defining lower and upper limits for each variable e.g. [[-3, 3], [-2, 2]]

    Returns:
    - Boolean: True if solution is within the constraints, False otherwise
    """
    if binary:
        x = np.array(decode_binary(x_bin, constraints, precision_digits))
    if len(constraints) == 0:
        return True  # No constraints provided, always return True
    
    for i in range(len(x)):
        if not (constraints[i][0] <= x[i] <= constraints[i][1]):
            return [a*np.sign(x_val) for a in constraints]
    return x_val

def print_verbose(verbose_level, x_best, fx_best, i):
    """
    For printing progress in search algorithms
    """
    if verbose_level == 1:
        print(f'i = {i}, ' + ', '.join([f'x{idx+1} = {val:.7f}' for idx, val in enumerate(x_best)]) + f', fx_best = {fx_best:.7f}', end='\r')
    elif verbose_level == 2:
        print(f'i = {i}, ' + ', '.join([f'x{idx+1} = {val:.7f}' for idx, val in enumerate(x_best)]) + f', fx_best = {fx_best:.7f}')

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

def encode_binary(real_value, constraint, precision_digits=4):
    """
    Encodes a real value back into binary format.
    
    Parameters:
    - real_value (float): The real value to encode.
    - constraint (tuple): Defining lower and upper limits for the real variable.
    - precision_digits (int): Precision digits for encoding.
    
    Returns:
    - binary_str (str): The encoded binary string.
    """
    low, high = constraint
    # Normalize the real value back to [0, 1] range
    max_value = 2**precision_digits - 1
    normalized_value = int((real_value - low) / (high - low) * max_value)
    # Convert normalized value to binary string
    binary_str = format(normalized_value, f'0{precision_digits}b')  # Format as a binary string
    return binary_str

def decode_binary(binary_str_list, constraints, precision_digits=4):
    """
    Decodes a single binary string to its corresponding real value based on constraints.

    Parameters:
    
    - binary_str (str): Encoded binary string for a single variable.
    - constraint (tuple): (low, high) limits for the real variable.
    - precision_digits (int): Number of decimal places of precision.

    Returns:
    - float: Decoded real value."""
    decoded_list = []
    for binary_str, constraint in zip(binary_str_list,constraints):
        low, high = constraint  # Unpack the constraint tuple
        integer_value = int(binary_str, 2)
        max_integer_value = (2 ** len(binary_str)) - 1
        normalized_value = integer_value / max_integer_value
        decoded = low + (high - low) * normalized_value
        decoded_list.append(round(decoded, precision_digits))
    return decoded_list
    
    
def decode_population(binary_array, constraints, precision_digits=4):
    """
    Takes a numpy array of binary strings and decodes into a numpy array of real values
    
    Parameters:
    - binary_array (np.array): Array of binary-encoded variables
    - constraints (list of tuples): Defining lower and upper limits for the real variable.
    - precision_digits (int): number of digits of precision if the representation is binary.
    
    Returns:
    np.array: Decoded real values for the entire population
    """
    decoded_population = np.zeros_like(binary_array)
    
    for i, binary_strs in enumerate(binary_array):
        decoded_array = [] 
        decoded_value = decode_binary(binary_strs, constraints, precision_digits)
        decoded_population[i] = decoded_value
    
    return np.array(decoded_population)

def binary_search(arr, target, low=0, high=None):
    """
    Given an np.array and a number, it finds the place for the number

    Parameters:
    - arr (np.array): sorted array
    - target (float): value searched
    - low (int): lower search boundary
    - high (int): higher search boundary

    Returns:
    - index (int): insertion position for the target
    """
    if high is None:
        high = len(arr) - 1
    
    # Base case: when the search range is exhausted, return the insertion position
    if low > high:
        return low  # This is where the target should be inserted
    
    mid = (low + high) // 2
    
    if arr[mid] == target:
        return mid  # If the target is found, return its position (you can modify this to always return insertion point if desired)
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, high)  # Search in the right half
    else:
        return binary_search(arr, target, low, mid - 1)  # Search in the left half

def spread_factor(u=None, nc=2):
    """
    Computes the spread factor for Simulated Binary Crossover (SBX) given the u value

    Parameters:
    - u (float): random u value from 0 to 1
    - nc (int): n_c value, n=0 uniform distribution, 2<n<5 matches closely the simulation for single-point crossover

    Returns:
    - beta (float)
    """
    if u is None:
        u = random.random()

    if u <= 0.5:
        beta = (2 * u) ** (1/(nc + 1))
    else:
        beta = (1/(2 * (1 - u))) ** (1/(nc + 1))
    return beta

def beta_q_factor(delta, eta_m, u=None):
    """
    Computes the beta_q factor for parameter based mutation (PM)

    Parameters:
    - delta (float): value calculated with y parent solution, y upper and lower limits.
    - eta_m (float): 100 + generation number aka η_m
    
    Returns:
    - Returns:
    - beta_q (float)
    """    
    if u is None:
        u = random.random()
    
    if u <= 0.5:
        delta_q = (2 * u + (1 - 2 * u) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
    else:
        delta_q = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta_m + 1)) ** (1/(eta_m + 1))

    return delta_q