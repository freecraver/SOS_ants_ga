import os
import time
from datetime import datetime
import pandas as pd

from knapsack.ga.ga_knapsack import solve_knapsack
from knapsack.aco.aco_knapsack import solve_aco_knapsack


INSTANCE_PATH = "res"
STATS_PATH = "stats"
NR_ITERATION = 10 # number of times a single instance should be evaluated (random fluctuations, confidence intervals,..)

def solve_ga(capacity, instances):
    res, _ = solve_knapsack(capacity, instances)
    best = res.keys[0].values
    return best[1]

def solve_aco(capacity, instances):
    res = solve_aco_knapsack(capacity, instances, ant_count=10, iteration_count=50)
    return res["fitness"]

def solve_instance(instance_name, capacity, instances):
    stats_lst = []

    for run in range(NR_ITERATION):
        for solver in [
            {"solver_name": "GA",
             "solve":lambda capacity, instances:  solve_ga(capacity, instances)},
            {"solver_name": "ACO",
             "solve": lambda capacity, instances: solve_aco(capacity, instances)}
        ]:
            print(f"Starting attempt {run + 1} for {solver['solver_name']}")
            start_time = time.time()
            best_value = solver["solve"](capacity, instances)
            exec_time = time.time() - start_time
            print(f"Best solution gives a value of {best_value}")
            stats_lst.append({
                "instance": instance_name,
                "method": solver["solver_name"],
                "value": best_value,
                "execution_time": exec_time,
                "run": run + 1})

    return stats_lst

def solve_folder(folder_name):
    print("*"*60)
    stats_lst = []
    print(f"Solving instances from folder {folder_name}...")
    for f in os.listdir(os.path.join(INSTANCE_PATH, folder_name)):
        print("-"*60)
        print(f"Solving instance {f}...")
        capacity, instances = load_instance(os.path.join(INSTANCE_PATH, folder_name, f))
        stats_lst.extend(solve_instance(f, capacity, instances))

    store_results(folder_name, stats_lst)

    print("*" * 60)


def store_results(folder_name, stats_lst):
    res_file_name = os.path.join(STATS_PATH, folder_name+"_run"+datetime.now().strftime('%m-%d_%H_%M_%S')+".csv")
    pd.DataFrame.from_dict(stats_lst).to_csv(res_file_name, index=False)


def load_instance(path):
    """

    :param path: path to instance file
    :return: capacity: float indicating capacity of knapsack,
            items: list of weight[0], value [1] items
    """
    with open(path, "r") as f:
        _, capacity = f.readline().split()
        item_infos = f.readlines()
        # read every single line and switch weight+value
        items = [list(map(float,item.split()[::-1])) for item in item_infos if len(item.split()) == 2]

    return float(capacity), items


if __name__ == "__main__":
    solve_folder("large_scale")