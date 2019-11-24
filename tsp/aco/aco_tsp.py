# this implementation was self-developed
# we therefor are not 100% able to ensure competitiveness to the big DEAP library :-(
# ps: this is not how I write python code at work :-)

import acopy
import tsplib95

def solve_aco_tsp(resPath):

    print(resPath)
    solver = acopy.Solver(rho=.03, q=.5)
    colony = acopy.Colony(alpha=1, beta=3)

    problem = tsplib95.load_problem(resPath)
    G = problem.get_graph()

    tour = solver.solve(G, colony, limit=100)

    return tour