import glob
import time
from collections import defaultdict
from statistics import mean, stdev
import sys, os, re
import pandas as pd
import random 
import threading
from sklearn.cluster import KMeans
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
TIME_TO_CAL = 10
K= 5
NUM_ITERATIONS = 5

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
    cycle_length
)

from lab2 import (
    plot_multiple_solutions

)
from lab3 import (
    lm_local_search
)

from lab4 import (
    evaluate_length,
    perturb_cycles
)


def heartbeat(interval=10):
    while True:
        print(f"[{time.strftime('%H:%M:%S')}] Heartbeat...")
        time.sleep(interval)


def nearest_neighbor_path(nodes, distance_matrix):
    if not nodes:
        return []

    start = random.choice(nodes)
    path = [start]
    unvisited = set(nodes)
    unvisited.remove(start)

    current = start
    while unvisited:
        next_node = min(unvisited, key=lambda node: distance_matrix[current][node])
        path.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return path

def initialize_population_knn(n, distance_matrix, coordinates, population_size=20):
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(coordinates)
    labels = kmeans.labels_

    cluster1 = [i for i in range(n) if labels[i] == 0]
    cluster2 = [i for i in range(n) if labels[i] == 1]

    population = []
    solution_lengths = []

    while len(population) < population_size:
        random.shuffle(cluster1)
        random.shuffle(cluster2)

        cycle1 = nearest_neighbor_path(cluster1, distance_matrix)
        cycle2 = nearest_neighbor_path(cluster2, distance_matrix)

        total_length = cycle_length(cycle1, distance_matrix) + cycle_length(cycle2, distance_matrix)

        if total_length not in solution_lengths:
            population.append((cycle1, cycle2, total_length))
            solution_lengths.append(total_length)

    return population, solution_lengths

def ILS_LNS(distance_matrix,coordinates, n, mode="ILS"):

    population, _ = initialize_population_knn(n, distance_matrix, coordinates, population_size=1)
    best_cycle1, best_cycle2, _ = population[0]
    start_time = time.time()
    ILS_num_itter = 0
    best_length = float("inf")

    while time.time() - start_time < TIME_TO_CAL:

        if mode == "ILS":
            cycle1, cycle2 = perturb_cycles(best_cycle1, best_cycle2, K)
            cycle1, cycle2 = lm_local_search(cycle1, cycle2, distance_matrix)

        ILS_num_itter += 1
        best_length, best_cycle1, best_cycle2 = evaluate_length(
            best_length, best_cycle1, best_cycle2, cycle1, cycle2, distance_matrix
        )

    return best_cycle1, best_cycle2, ILS_num_itter



def process_file(file_paths, algorithms, num_iterations=1):
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
        print(f"Processing: {file_path}")
        distance_matrix, coordinates, n = read_instance(file_path)
        for algorithm in algorithms:
            config_key = f"{algorithm}"

            for i in range(num_iterations):
                print(config_key, i)
                start_time = time.time()
                cycle1, cycle2, num_i = ILS_LNS(
                                distance_matrix, coordinates , n, algorithm
                            )

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
                    ] = f"{algorithm}"

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
    algorithms = ["ILS"]
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))

    hb_thread = threading.Thread(target=heartbeat, args=(30,), daemon=True)
    hb_thread.start()

    results = process_file(file_paths, algorithms, NUM_ITERATIONS)
