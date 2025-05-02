import glob
import time
from collections import defaultdict, deque
import sys, os, re
import pandas as pd
import random 
import threading
MSLS_TRY_NUM = 200
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

for name in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, name)
    if (
        re.match(r"lab\d+", name)
        and os.path.isdir(full_path)
        and full_path != current_dir
    ):
        sys.path.append(full_path)

from lab1 import (
    read_instance,
    cycle_length,
)

from lab2 import (
    random_cycles,
    plot_multiple_solutions,
)
from lab3 import (
    lm_local_search
)

def heartbeat(interval=10):
    while True:
        print(f"[{time.strftime('%H:%M:%S')}] Heartbeat...")
        time.sleep(interval)

def perturb_cycles(cycle1, cycle2, k):
    cycle1, cycle2 = cycle1.copy(), cycle2.copy()
    indices1 = random.sample(range(len(cycle1)), k)
    indices2 = random.sample(range(len(cycle2)), k)

    for i in range(k):
        cycle1[indices1[i]], cycle2[indices2[i]] = cycle2[indices2[i]], cycle1[indices1[i]]
    
    return cycle1, cycle2

def evaluate_length(best_length,best_cycle1,best_cycle2,cycle1,cycle2,distance_matrix):
    length = cycle_length(cycle1, distance_matrix) + cycle_length(cycle2, distance_matrix)
    if length < best_length:
            print(length)
            return length,cycle1,cycle2
    
    return best_length,best_cycle1,best_cycle2

def MSLS(distance_matrix,n):

    best_length = float('inf')
    best_cycle1 = None
    best_cycle2 = None

    for i in range(MSLS_TRY_NUM):
        cycle1, cycle2 = random_cycles(n)
        cycle1, cycle2 = lm_local_search(cycle1, cycle2, distance_matrix)
        best_length,best_cycle1,best_cycle2 = evaluate_length(best_length,best_cycle1,best_cycle2,cycle1,cycle2,distance_matrix)

    return best_cycle1,best_cycle2

def ILS(distance_matrix,n,time_to_search):

    best_length = float('inf')
    best_cycle1, best_cycle2 = random_cycles(n)
    start_time = time.time()
    ILS_num_itter = 0

    while time.time() - start_time < time_to_search:
        cycle1, cycle2 = perturb_cycles(best_cycle1,best_cycle2,5)
        cycle1, cycle2 = lm_local_search(cycle1, cycle2, distance_matrix)
        ILS_num_itter+=1
        best_length,best_cycle1,best_cycle2 = evaluate_length(best_length,best_cycle1,best_cycle2,cycle1,cycle2,distance_matrix)
    return best_cycle1,best_cycle2, ILS_num_itter


def process_file(file_paths, init_methods, modes, algorithms, num_iterations=1):
    for file_path in file_paths:
        results = defaultdict(
            lambda: {
                "result_list": [],
                "time": [],
                "num_i": [],
                "best_result": float("inf"),
                "best_cycle1": None,
                "best_cycle2": None,
            }
        )
        best_solutions = []
        time_for_other =0
        print(f"Processing: {file_path}")
        distance_matrix, coordinates, n = read_instance(file_path)
        for init_method in init_methods:
            for mode in modes:
                for algorithm in algorithms:
                    config_key = f"{init_method}-{mode}-{algorithm}"

                    for i in range(num_iterations):
                        print(config_key, i)
                        num_i = MSLS_TRY_NUM
                        start_time = time.time()
                        if algorithm == "MSLS":
                            cycle1, cycle2 = MSLS(distance_matrix,n)
                            time_for_other = time.time() - start_time
                        elif algorithm == "ILS":
                            cycle1, cycle2, num_i = ILS(distance_matrix,n,time_for_other)
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        print(f"Elapsed time: {elapsed_time:.2f} seconds")

                        length = cycle_length(cycle1, distance_matrix) + cycle_length(
                            cycle2, distance_matrix
                        )
                        print(length)

                        results[config_key]["result_list"].append(length)
                        results[config_key]["time"].append(elapsed_time)
                        results[config_key]["num_i"].append(num_i)

                        # Check if it's the best result
                        if length < results[config_key]["best_result"]:
                            results[config_key]["best_result"] = length
                            results[config_key]["best_cycle1"] = cycle1
                            results[config_key]["best_cycle2"] = cycle2
                            results[config_key][
                                "name"
                            ] = f"{init_method}-{mode}-{algorithm}"

                    # Add best solution for plotting
                    best_cycle1 = results[config_key]["best_cycle1"]
                    best_cycle2 = results[config_key]["best_cycle2"]
                    name = results[config_key]["name"]
                    best_solutions.append((best_cycle1, best_cycle2, coordinates, name))

        f = os.path.basename(file_path)
        f = f.split(".")[0]
        plot_multiple_solutions(best_solutions, f)

        save_best_results(results, f)

    return results


def save_best_results(results, file_name):

    output = []

    for config_key, config_results in results.items():

        num_iterations = len(config_results["result_list"])
        avg_result = sum(config_results["result_list"]) / num_iterations
        min_result = min(config_results["result_list"])
        max_result = max(config_results["result_list"])
        avg_time = sum(config_results["time"]) / num_iterations
        min_time = min(config_results["time"])
        max_time = max(config_results["time"])
        avg_num_i = sum(config_results["num_i"]) / num_iterations
        row = [
            config_key,
            avg_result,
            min_result,
            max_result,
            avg_time,
            min_time,
            max_time,
            avg_num_i
        ]
        output.append(row)

    df = pd.DataFrame(output, columns=["Configuration", "avg_result", "min_result", "max_result","avg_time","min_time","max_time","avg_num_i"])
    df.to_csv(f"results_{file_name}.csv", index=False)


if __name__ == "__main__":
    folder_path = "../data/"
    init_methods = ["rand"]
    modes = ["edges"]
    algorithms = ["MSLS","ILS"]
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))
    num_iterations = 1

    hb_thread = threading.Thread(target=heartbeat, args=(30,), daemon=True)
    hb_thread.start()

    results = process_file(file_paths, init_methods, modes, algorithms, num_iterations)
