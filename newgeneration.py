import numpy as np
import random
from boid import Boid

def choose_offspring(fitness_values):
    # Calculate the total fitness
    total_fitness = sum(fitness_values)
    n_draws = len(fitness_values)
    
    # Calculate the cumulative fitness
    cumulative_fitness = [sum(fitness_values[:i+1]) for i in range(len(fitness_values))]
    selected_individuals = np.zeros(n_draws, dtype = int)
    
    for index in range(n_draws):
        # Generate a random number between 0 and the total fitness
        r = random.uniform(0, total_fitness)
        
        # Select the individual where the random number falls within their cumulative fitness range
        for i, cum_fit in enumerate(cumulative_fitness):
            if r <= cum_fit:
                selected_individuals[index] = i
                break
                
    return selected_individuals

def update_flock(flockMates, params):
    parents = flockMates.copy()
    fitnesses = np.array([mate.fitness for mate in flockMates])
    offspring_indeces = choose_offspring(fitnesses)

    for offspring_index,parent_index in enumerate(offspring_indeces):
        parent = parents[parent_index]
        traits = {"radius":parent.radius,"coop":parent.coop}
        flockMates[offspring_index].offspring_gen(traits,params)