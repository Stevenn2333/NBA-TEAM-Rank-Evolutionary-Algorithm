"""
My collection of survivor selection methods

Student number: 20143750
Student name:Feiting Yang
"""

import random


def mu_plus_lambda(population,current_fitness, offspring,offspring_fitness):
    # Combine population and offspring
    combined_population = population + offspring
    mu = len(population)
    fitness = current_fitness + offspring_fitness
    sorted_population = sorted(zip(combined_population, fitness), key=lambda x: x[1])
    
    # Select the mu best individuals
    next_population = [ind for (ind, fit) in sorted_population[:mu]]
    
    return next_population


def replacement(current_pop, current_fitness, offspring, offspring_fitness):
    """replacement selection"""

    #population = []
    #fitness = []
    sorted_cur_population = sorted(zip(current_pop, current_fitness), key=lambda x: x[1])
    sorted_off_population = sorted(zip(offspring, offspring_fitness), key=lambda x: x[1])
    sorted_population = sorted_cur_population+sorted_off_population
    lambda1 = len(offspring)
    next_population = [ind for (ind, fit) in sorted_population[lambda1:]]
    return next_population

    


def random_uniform(current_pop, current_fitness, offspring, offspring_fitness):
    """random uniform selection"""
    combined_population = current_pop + offspring
    mu = len(current_pop)
    fitness = current_fitness + offspring_fitness
    sorted_population = sorted(zip(combined_population, fitness), key=lambda x: x[1])
    next_population = [ind for (ind, fit) in random.sample(sorted_population,mu)]
    return next_population
    '''population = []
    fitness = []

    # student code starts
    mu = len(current_pop)
    lambda1 = len(offspring)
    curr = []
    for i in range(mu):
        curr.append((current_pop[i],current_fitness[i]))
    curr = sorted(curr)
    off = []
    for j in range(lambda1):
        off.append((offspring[j],offspring_fitness[j]))
    total = curr + off
    result = random.sample(total,mu)
    for ix in range(mu):
        population.append(result[ix][0])
        fitness.append(result[ix][1])

    
    # student code ends
    
    return population, fitness'''
'''current_pop = ['a','b','c','d','e','f','g']
current_fitness=[1,2,3,4,5,6,7]
offspring = ['x','y','z']
offspring_fitness = [10,11,12]
#print(mu_plus_lambda(current_pop, current_fitness, offspring, offspring_fitness))
#print(replacement(current_pop, current_fitness, offspring, offspring_fitness))
print(random_uniform(current_pop, current_fitness, offspring, offspring_fitness))
'''
