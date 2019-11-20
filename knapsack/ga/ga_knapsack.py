#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

# comment group 8: please note that we modified this file for the purpose of the exercise
# the original version is availabe at the deap github repo

import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

IND_INIT_SIZE = 5

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

def solve_knapsack(capacity, items, mu = 50, lambd = 100, cxpb = 0.7, mutpb = 0.2, generation_cnt=50, random_seed=None):
    """
    :param capacity: Max capacity of the knapsack
    :param items: list of items, of form [weight, value]
    :param lambd: The number of children to produce at each generation.
    :param mu: The number of individuals to select for the next generation.
    :param generation_cnt: number of generations
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param random_seed: may be used to get reproducable results
    :return: ParetFront, whereas items is a list of solutions (~iterations) and keys a list of Fitness objects
                The fitness objects can be used to determine fitness scores for every iteration
    """

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, len(items))

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, IND_INIT_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalKnapsack(individual):
        weight = 0.0
        value = 0.0
        for item in individual:
            weight += items[item][0]
            value += items[item][1]
        if weight > capacity:
            return 1000000, 0  # Ensure overweighted bags are dominated
        return weight, value

    def cxSet(ind1, ind2):
        """Apply a crossover operation on input sets. The first child is the
        intersection of the two sets, the second child is the difference of the
        two sets.
        """
        temp = set(ind1)  # Used in order to keep type
        ind1 &= ind2  # Intersection (inplace)
        ind2 ^= temp  # Symmetric Difference (inplace)
        return ind1, ind2

    def mutSet(individual):
        """Mutation that pops or add an element."""
        if random.random() < 0.5:
            if len(individual) > 0:  # We cannot pop from an empty set
                individual.remove(random.choice(sorted(tuple(individual))))
        else:
            individual.add(random.randrange(len(items)))
        return individual,

    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)

    if random_seed:
        random.seed(random_seed)

    pop = toolbox.population(n=mu)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu, lambd, cxpb, mutpb, generation_cnt, stats,
                              halloffame=hof)

    return hof, stats
