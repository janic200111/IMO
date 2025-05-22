import glob
import time
from collections import defaultdict
from statistics import mean, stdev
import sys, os, re
import pandas as pd
import random 
import threading
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
TIME_TO_CAL = 188.37
POP_SIZE = 20
BASE_MUTATION_RATE = 0.2
K= 5

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
    heuristic_algorithm_regret_weighted,
    greedy_algorithm_nearest_neighbour
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

def recombine(parent1, parent2, n, distance_matrix):
    p1_cycle1, p1_cycle2 = parent1
    p2_cycle1, p2_cycle2 = parent2

    def extract_edges(cycle):
        return set((cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle)))

    edges_p2 = extract_edges(p2_cycle1).union(extract_edges(p2_cycle2))

    def filtered_cycle(cycle):
        new_cycle = []
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            if (a, b) in edges_p2 or (b, a) in edges_p2:
                new_cycle.append(a)
        return new_cycle

    child1 = filtered_cycle(p1_cycle1)
    child2 = filtered_cycle(p1_cycle2)
    
    visited = set(child1 + child2)

    child1, child2 = heuristic_algorithm_regret_weighted(
        distance_matrix,
        n,
        cycle1=child1,
        cycle2=child2,
        visited=visited
    )

    return child1, child2


def initialize_population(n, distance_matrix):
    population = []
    solution_lengths = []

    while len(population) < POP_SIZE:
        cycle1, cycle2 = random_cycles(n)
        cycle1, cycle2 = lm_local_search(cycle1, cycle2, distance_matrix)
        value = cycle_length(cycle1, distance_matrix) + cycle_length(cycle2, distance_matrix)

        if value not in solution_lengths:
            population.append((cycle1, cycle2, value))
            solution_lengths.append(value)

    return population, solution_lengths

def HAE(distance_matrix, n, with_local_search=True,mut=True):
    start_time = time.time()
    population, solution_lengths = initialize_population(n, distance_matrix)
    HAE_num_itter = 0
    mutation_chance = 0
    best_length = float('inf')

    while time.time() - start_time < TIME_TO_CAL:
        parent1, parent2 = random.sample(population, 2)
        child1, child2 = recombine((parent1[0], parent1[1]), (parent2[0], parent2[1]), n, distance_matrix)

        if mut:
            mean_fit = mean(solution_lengths)
            std_fit = stdev(solution_lengths)
            within_std = [
                l for l in solution_lengths
                if abs(l - mean_fit) <= std_fit
            ]
            mutation_chance = 1 - len(within_std) / len(solution_lengths)
            if random.random() < mutation_chance:
                child1, child2 = perturb_cycles(child1, child2, K)

        if with_local_search:
            child1, child2 = lm_local_search(child1, child2, distance_matrix)

        value = cycle_length(child1, distance_matrix) + cycle_length(child2, distance_matrix)

        if best_length > value:
            print(value)
            print(HAE_num_itter)
            best_length = value

        if value not in solution_lengths:
            worst = max(population, key=lambda x: x[2])
            if value < worst[2]:
                population.remove(worst)
                population.append((child1, child2, value))
                solution_lengths.remove(worst[2])
                solution_lengths.append(value)
        HAE_num_itter += 1

    # Zwracamy najlepsze rozwiązanie
    best = min(population, key=lambda x: x[2])
    return best[0], best[1], HAE_num_itter


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
                if algorithm == "HAE":
                    cycle1, cycle2, num_i = HAE(distance_matrix,n,mut=False)
                elif algorithm == "HAE-M":
                    cycle1, cycle2, num_i = HAE(distance_matrix,n,mut=True)
                elif algorithm == "HAEa":
                    cycle1, cycle2, num_i = HAE(distance_matrix,n,False,False)
                else:
                    cycle1, cycle2, num_i = HAE(distance_matrix,n,False,True)

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
    algorithms = ["HAE","HAEa","HAE-M","HAEa-M"]
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))
    num_iterations = 1

    hb_thread = threading.Thread(target=heartbeat, args=(30,), daemon=True)
    hb_thread.start()

    results = process_file(file_paths, algorithms, num_iterations)
