from functions import *

########################
## Genetic Algorithms ##
########################

def ga_solve(mu,binary,f_x, constraints, minimization, n_generations = 100, pc=0.9, precision_digits=4):
    """

    Parameters:
    - mu (int): number of individuals/chromosomes of the GA.
    - f_x (sympy exp): Objective function or list of derivates.
    - binary (boolean): whether the chromosomes are binary or real.
    - constraints (list of tuples): Defining lower and upper limits for each variable.
    - minimization (boolean): whether the f_x is a minimization problem.
    - n_generations (int): number of generations
    - pc (float): probability of crossover in binary
    """
    sorted_symbols = sorted(f_x.free_symbols, key=lambda s: s.name)
    n_genes = len(sorted_symbols)
    p_m = 1/mu
    t = 0 # generation

    population = list() # List of individuals/chromosomes 
    for i in range(mu):
        individual = initialize(n_genes, binary, constraints=constraints) # np.arrays
        population.append(individual)
    
    avg_fitness_list = []
    std_fitness_list = []
    max_fitness_list = []
    
    while t < n_generations:
        population_fitness = eval_population(population, f_x, binary, constraints, precision_digits)
        avg_fitness_list.append(np.mean(population_fitness))
        std_fitness_list.append(np.std(population_fitness))
        max_fitness_list.append(np.min(population_fitness))
        selected_population = list()
        if binary: # Wheel selection
            for i in range(mu):
                individual = roulete_wheel(population,f_x,constraints=constraints,minimization=minimization)
                selected_population.append(individual)

            shuffled_indices = np.random.permutation(mu) # 1, 2, 3, 4 --> 2, 4, 1, 3
            selected_population = [selected_population[i] for i in shuffled_indices]

            list_children = []

            for i in range(0, mu, 2): # 1, 3
                parent1, parent2 = selected_population[i], selected_population[i+1]

                QC = False
                while QC != True:
                    filter_crossover = random.random()
                    if filter_crossover < pc:
                        child1, child2 = point_crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1, parent2

                    QC = constraint_checker(child1,constraints,binary, precision_digits) and constraint_checker(child2,constraints,binary, precision_digits)

                child1_gene = np.random.randint(0, len(child1)) # To be mutated  
                child1[child1_gene] = binary_mutation(child1[child1_gene], p = p_m)

                child2_gene = np.random.randint(0, len(child2)) # To be mutated
                child1[child2_gene] = binary_mutation(child2[child2_gene], p = p_m)

                #child1, child2 = constraint_checker(child1, constraints, binary), constraint_checker(child2, constraints, binary)

                list_children.append(child1)
                list_children.append(child2)

        else: # Tournament selection (ran twice to keep the number of individuals the same)
            winners_1 = tournament_selection(population, f_x, binary=binary, minimization=minimization) # Half of the individuals selected
            winners_2 = tournament_selection(population, f_x, binary=binary, minimization=minimization) # Half of the individuals selected
            selected_population = winners_1 + winners_2

            list_children = []

            for i in range(0, mu, 2): # 1, 3
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = point_crossover(parent1, parent2)

                filter_crossover = random.random()
                if filter_crossover < pc:
                    child1, child2 = sbx(parent1, parent2, nc=20)
                else:
                    child1, child2 = parent1, parent2

                child1 = parameter_based_mutation(child1,constraints=constraints,t=t)
                child2 = parameter_based_mutation(child1,constraints=constraints,t=t)

                child1, child2 = constraint_checker(child1, constraints), constraint_checker(child2, constraints)

                list_children.append(child1)
                list_children.append(child2)
        
        

        population = list_children
        t += 1
        print(f"{t}/{n_generations} done", end='\r')
    return avg_fitness_list, std_fitness_list, max_fitness_list