import random
import pandas as pd
import ast
import math
import time


def read_data(filename):
    df = pd.read_csv(filename)
    return df

#F: formula F O F, F 0 N， N O F， N O N， V O V， V O F, F O V, V O N, N O V. N: Numeric value, V: Variable, O: operator
class FormulaEvaluator:
    def __init__(self, data):
        self.data = data

    def evaluate(self, formula):
        tree = ast.parse(formula, mode='eval')
        return self._eval_node(tree.body)

    def _eval_node(self, node):
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            operator = self._eval_node(node.op)
            return operator(left, right)
        elif isinstance(node, ast.Name):
            return self.data.get(node.id)
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Add):
            return lambda x, y: x + y
        elif isinstance(node, ast.Sub):
            return lambda x, y: x - y
        elif isinstance(node, ast.Mult):
            return lambda x, y: x * y
        elif isinstance(node, ast.Div):
            return lambda x, y: x / y
        else:
            raise ValueError(f"Invalid node type: {node}")


def print_formula(formula):
    terms = formula.split()
    operators = terms[1::2]
    expression = ''

    for i in range(len(terms)):
        expression += terms[i]
        if i % 2 == 1:
            expression += f' {operators[i // 2]} '
    print(expression)

def generate_random_expression():
    operators = ['+', '-', '*', '/']
    _expression = ''

    for _ in range(3):
        term1 = random.choice(terms)
        term2 = random.choice(terms)
        operator = random.choice(operators)

        _expression += f'({term1} {operator} {term2})'
        if _ < 2:
            _expression += f'{random.choice(operators)}'

    return _expression


def evaluate_expression(expression,row):
    '''try:
        result = eval(expression, {}, row.to_dict())
    except ZeroDivisionError:
        result = float('inf')'''
    try:
        expression = generate_random_expression()
        data = {term: random.uniform(0, 100) for term in terms}
        evaluator = FormulaEvaluator(data)
        #print(data)
        result = evaluator.evaluate(expression)
        #print(result)
    except ZeroDivisionError:
        result = float('inf')
    #print(result)
    return result


def compute_fitness(population, dict):
    fitness_values = []

    for individual in population:
        total_difference = 0
        for row in range(len(dict)):
            result = evaluate_expression(individual, dict[row])
            #print(dict[row]['Rank']-result)
            total_difference += abs(dict[row]['Rank'] - result)
            #print(total_difference)
        fitness_values.append(1 / total_difference)
        #print(total_difference/len(dict))
        #print(total_difference)
        #fitness_values.append(total_difference/len(dict))
        #print(fitness_values)

    return fitness_values


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual):
    index = random.randint(0, len(individual) - 1)
    mutated_individual = individual[:index] + generate_random_expression() + individual[index + 1:]
    return mutated_individual


def select(population, fitness_values):
    selected_indices = random.choices(range(len(population)), weights=fitness_values, k=2)
    return [population[i] for i in selected_indices]

terms = ['Teams','Games', 'MinutesPlayed', 'FieldGoals', 'FieldGoalAttempts', 'FieldGoalPercentage', 'ThreePointers',
         'ThreePointAttempts', 'ThreePointPercentage', 'TwoPointers', 'TwoPointAttempts', 'TwoPointPercentage',
         'FreeThrows', 'FreeThrowAttempts', 'FreeThrowPercentage', 'OffensiveRebounds', 'DefensiveRebounds',
         'TotalRebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'PersonalFouls', 'Points', 'year', 'Rank']

def main():
    start = time.time()
    data = read_data('output.csv')
    num_generations = 100
    population_size = 100
    crossover_rate = 0.7
    mutation_rate = 0.1
    depth = 3

    def create_dicts(dataframe):
        dicts = []
        for _, row in dataframe.iterrows():
            row_dict = {}
            for column_name in dataframe.columns:
                row_dict[column_name] = row[column_name]
            dicts.append(row_dict)
        return dicts

    # put the row in a dictionary
    dictionaries = create_dicts(data)
    #print(dictionaries[0])

    population = [generate_random_expression() for _ in range(population_size)]

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
    print('Fitness value of Best individual:',max(fitness_values))

    #Evaluation
    difference = 0
    results = []
    for i in range(len(dictionaries)):
        result = evaluate_expression(best_individual, dictionaries[i])
        results.append((i, result))

    results = []
    for i in range(len(dictionaries)):
        result = evaluate_expression(best_individual, dictionaries[i])
        results.append((i, result))

    # Sort the results based on the result value (second item in the tuple)
    sorted_results = sorted(results, key=lambda x: x[1])

    # Match the ranks with the result and the corresponding i
    rank_to_i = {}
    for rank, (i, result) in enumerate(sorted_results, 1):
        rank_to_i[rank] = i

    # Create an inverse mapping of rank_to_i
    i_to_rank = {i: rank for rank, i in rank_to_i.items()}  

    print("i to rank is ",i_to_rank)

    team_names = [
    "Boston Celtics", "Dallas Mavericks", "Golden State Warriors", "Miami Heat",
    "Cleveland Cavaliers", "New York Knicks", "Toronto Raptors", "Phoenix Suns",
    "Philadelphia 76ers", "Utah Jazz", "Los Angeles Clippers", "Memphis Grizzlies",
    "New Orleans Pelicans", "Denver Nuggets", "Oklahoma City Thunder", "Washington Wizards",
    "Chicago Bulls", "Milwaukee Bucks", "Brooklyn Nets", "Orlando Magic",
    "Atlanta Hawks", "Detroit Pistons", "San Antonio Spurs", "Minnesota Timberwolves",
    "Indiana Pacers", "Charlotte Hornets", "Los Angeles Lakers", "Portland Trail Blazers",
    "Sacramento Kings", "Houston Rockets", "League Average"
    ]

    # Replace the keys with the corresponding team names
    i_to_team = {i: team_names[i] for i in range(len(team_names))}

    # Use the correct i_to_rank mapping
    rank_to_team = {rank: i_to_team[i] for i, rank in i_to_rank.items()}

    print("Rank to team is:", rank_to_team)



    for i in range(len(dictionaries)):
        difference += (dictionaries[i]['Rank'] - i_to_rank.get(i))**2
    MSE = difference/len(dictionaries)  
    print('The MSE of best individual is:',MSE) 

    end = time.time()
    print('Running time:', end - start, 'seconds')

if __name__ == '__main__':
    main()
