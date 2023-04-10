"""
My collection of parent selection methods

Student number:20143750
Student name:Feiting Yang
"""

import random




def MPS(fitness, mating_pool_size,population):
    """Multi-pointer selection (MPS)"""

    selected_to_mate = []
    fitness_p = cumulative_prob_distribution(fitness)
    step = 1/mating_pool_size
    r = random.uniform(0,step)
    current_member = 0
    i = 0
    while current_member < mating_pool_size:
        while r <= fitness_p[i]:
            selected_to_mate.append(i)
            r += step
            current_member += 1
        i += 1
    return [population[i] for i in selected_to_mate]


def tournament(fitness, mating_pool_size, tournament_size,population):
    """Tournament selection without replacement"""

    selected_to_mate = []

    # student code starts
    current_number  = 0
    while current_number < mating_pool_size:
        # randomly pick 3 individual to compare   
        candidate = random.sample(range(tournament_size),2)
        winner = 0
        for j in candidate:
            if fitness[j] > winner:
                winner = fitness[j]# find the fittest one
        selected_to_mate.append(fitness.index(winner)) # the index of fitness list
        current_number += 1    
    return [population[i] for i in selected_to_mate]


def random_uniform (population_size, mating_pool_size,population):
    """Random uniform selection"""

    selected_to_mate = []
    i = 0
    while i < mating_pool_size:
        selected_to_mate.append(random.randint(0,population_size-1))
        i += 1
    return [population[i] for i in selected_to_mate]

def cumulative_prob_distribution(fitness):
    s = sum(fitness)
    result = []
    cp = 0
    if s == 0:
        for j in range(len(fitness)):
            cp += 1/len(fitness)
            result.append(cp)
    else:
        for i in fitness:
            cp += i/s
            result.append(cp)
    return result
#print(cumulative_prob_distribution([1,2,3,4]))
#print(MPS([1,2,3,4],4))
#print(random.sample([1,2,3,4,5,6,7,8],5))
#print(tournament([2,2,2,2,2,6,7],4,7))
#print(random_uniform(10,5))


