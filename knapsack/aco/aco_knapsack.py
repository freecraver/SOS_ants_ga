# this implementation was self-developed
# we therefor are not 100% able to ensure competitiveness to the big DEAP library :-(
# ps: this is not how I write python code at work :-)

import random

def solve_aco_knapsack(capacity, items, ant_count=50, iteration_count=100, min_trail=1e-7, random_seed=None, trace=True):
    """

    :param capacity: Max capacity of the knapsack
    :param items: list of items, of form [weight, value]
    :param ant_count: number of ants to be used
    :param iteration_count: number of iterations until process is stopped
    :param min_trail: minimum amount of pheromon trail (i.e. probability to choose an item)
    :param random_seed: may be used to get reproducible results
    :param trace: if true, detailed info is printed
    :return: dict with key "fitness" giving value of best solution and "items", giving chosen items
    """

    if random_seed:
        random.seed(random_seed)

    original_idx = sorted(range(len(items)), key=items.__getitem__)
    items = sorted(items, key=lambda x: x[0])

    # tau from slides, global amount of pheromones deposited
    trail = [1] * len(items) # all are equal at the start
    # utility of an item, the higher the ratio of value to weight, the more likely ants will choose it
    util = [i[1]/i[0] for i in items]
    util = [u/max(util) for u in util] # normalize

    best_solution = {"fitness": 0, "items": []}

    def get_transition_prob(item_idx, visited_idx, cap_to_use):
        if visited_idx.get(item_idx):
            return 0 # already included
        if items[item_idx][0] > cap_to_use:
            return 0 # no space left
        prob = trail[item_idx] * util[item_idx] # bias against items with low utilization
        return prob

    def run_iteration(best_solution):
        fitness_store = []  # store fitness for individual paths
        iter_best_solution = best_solution
        for ant_i in range(ant_count):
            solution = run_ant()
            if solution["fitness"] > iter_best_solution["fitness"]:
                # update new best solution
                if trace:
                    orig_idx = [original_idx[idx] for idx in solution["items"]]
                    print(f"Found new best solution with value {solution['fitness']} and items [{','.join(map(str,orig_idx))}]")
                iter_best_solution = solution
            fitness_store.append(solution)

        update_trails(fitness_store)
        return iter_best_solution

    def run_ant():
        visited_idx = {}
        cap_to_use = capacity
        while True:
            transition_probs = [get_transition_prob(idx, visited_idx, cap_to_use) for idx in range(len(items))]
            prob_sum = sum(transition_probs)

            if prob_sum == 0:
                # no fitting items left
                earned_value = sum([items[idx][1] for idx in visited_idx])
                return {"fitness": earned_value, "items": visited_idx}

            # normalize probabilities for efficiency
            transition_probs = [p / prob_sum for p in transition_probs]
            is_item_chosen = False
            while not is_item_chosen:
                # do this as long as one item is chosen - can be made more efficient of course - but E()=1
                for idx, p in enumerate(transition_probs):
                    if p > random.random():
                        # we choose item at idx
                        visited_idx[idx] = True
                        cap_to_use -= items[idx][0]
                        is_item_chosen = True
                        break

    def update_trails(fitness_store):
        """
        updates global pheromone trail based on global information
        :param fitness_store:
        :return:
        """
        fitness_sum = sum([sol["fitness"] for sol in fitness_store])
        for idx in range(len(fitness_store)):
            fitness = fitness_store[idx]["fitness"]/fitness_sum
            fitness = max([fitness, min_trail])
            trail[idx] = fitness

    for iter in range(iteration_count):
        best_solution = run_iteration(best_solution)
        return best_solution


