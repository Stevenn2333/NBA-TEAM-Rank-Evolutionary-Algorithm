import random
import pandas as pd
import ast
import math
import time


def read_data(filename):
    df = pd.read_csv(filename)
    return df


class ExprNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class FormulaEvaluator:
    def __init__(self, data):
        self.data = data

    def evaluate(self, node):
        return self._eval_node(node)

    def _eval_node(self, node):
        if node.left is None and node.right is None:
            return self.data.get(node.value)

        left = self._eval_node(node.left) if node.left is not None else None
        right = self._eval_node(node.right) if node.right is not None else None
        op = node.value

        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            try:
                return left / right
            except ZeroDivisionError:
                return float('inf')
        else:
            raise ValueError(f"Invalid node type: {node}")


def create_dicts(dataframe):
    dicts = []
    for _, row in dataframe.iterrows():
        row_dict = {}
        for column_name in dataframe.columns:
            row_dict[column_name] = row[column_name]
        dicts.append(row_dict)
    return dicts

terms = ['Teams', 'Games', 'MinutesPlayed', 'FieldGoals', 'FieldGoalAttempts', 'FieldGoalPercentage', 'ThreePointers',
             'ThreePointAttempts', 'ThreePointPercentage', 'TwoPointers', 'TwoPointAttempts', 'TwoPointPercentage',
             'FreeThrows', 'FreeThrowAttempts', 'FreeThrowPercentage', 'OffensiveRebounds', 'DefensiveRebounds',
             'TotalRebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'PersonalFouls', 'Points', 'year', 'Rank']
def random_expr():
    operators = ['+', '-', '*', '/']
    

    term1 = random.choice(terms)
    term2 = random.choice(terms)
    operator = random.choice(operators)

    return ExprNode(operator, ExprNode(term1), ExprNode(term2))


def evaluate_expression(expression, row):
    try:
        data = {term: row[term] for term in terms}
        evaluator = FormulaEvaluator(data)
        result = evaluator.evaluate(expression)
    except ZeroDivisionError:
        result = float('inf')
    return result


def compute_fitness(population, dictionaries):
    fitness_values = []

    for individual in population:
        total_difference = 0
        for row in dictionaries:
            result = evaluate_expression(individual, row)
            total_difference += abs(row['Rank'] - result)
        fitness_values.append(1 / total_difference)

    return fitness_values


def crossover(parent1, parent2):
    if parent1.left is not None and parent2.left is not None:
        child1_left, child2_left = crossover(parent1.left, parent2.left)
    else:
        child1_left, child2_left = parent1.left, parent2.left

    if parent1.right is not None and parent2.right is not None:
        child1_right, child2_right = crossover(parent1.right, parent2.right)
    else:
        child1_right, child2_right = parent1.right, parent2.right

    child1 = ExprNode(parent1.value, child1_left, child1_right)
    child2 = ExprNode(parent2.value, child2_left, child2_right)

    return child1, child2

def mutate(individual):
    if random.random() < 0.5 and individual.left is not None:
        individual.left = mutate(individual.left)
    elif individual.right is not None:
        individual.right = mutate(individual.right)
    else:
        individual = random_expr()
    return individual

def select(population, fitness_values):
    selected_indices = random.choices(range(len(population)), weights=fitness_values, k=2)
    return [population[i] for i in selected_indices]

def main():
    start = time.time()
    data = read_data('output.csv')
    num_generations = 100
    population_size = 100
    crossover_rate = 0.7
    mutation_rate = 0.1
    depth = 3
    dictionaries = create_dicts(data)

    population = [random_expr() for _ in range(population_size)]

    for generation in range(num_generations):
        fitness_values = compute_fitness(population, dictionaries)
        new_population = []

        for _ in range(population_size // 2):
            parents = select(population, fitness_values)

            if random.random() < crossover_rate:
                offspring = crossover(parents[0], parents[1])
            else:
                offspring = parents

            for child in offspring:
                if random.random() < mutation_rate:
                    child = mutate(child)
                new_population.append(child)
        population = new_population

    best_individual = population[fitness_values.index(max(fitness_values))]
    print('Best individual:', best_individual)
    print('Fitness value of Best individual:', max(fitness_values))

    # Evaluation
    difference = 0
    results = []
    for i in range(len(dictionaries)):
        result = evaluate_expression(best_individual, dictionaries[i])
        results.append((i, result))

    sorted_results = sorted(results, key=lambda x: x[1])

    rank_to_i = {}
    for rank, (i, result) in enumerate(sorted_results, 1):
        rank_to_i[rank] = i

    i_to_rank = {i: rank for rank, i in rank_to_i.items()}

    print("i to rank is ", i_to_rank)
    for i in range(len(dictionaries)):
        difference += (dictionaries[i]['Rank'] - i_to_rank.get(i)) ** 2
    MSE = difference / len(dictionaries)
    print('The MSE of best individual is:', MSE)

    end = time.time()
    print('Running time:', end - start, 'seconds')

if __name__ == '__main__':
    main()




