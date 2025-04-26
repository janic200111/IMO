import glob
import time
from collections import defaultdict, deque
import sys, os, re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

for name in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, name)
    if (re.match(r"lab\d+", name) and os.path.isdir(full_path) and full_path != current_dir):
        sys.path.append(full_path)
from lab1 import (
    read_instance,
    heuristic_algorithm_regret_weighted,
    cycle_length,
)

from lab2 import (
    swap_in_cycle_nodes,
    swap_in_cycle_edges,
    swap_between_cycles,
    local_search,
    random_cycles,
    save_best_results,
    plot_multiple_solutions
)
def lm_local_search(cycle1, cycle2, distance_matrix):

    def evaluate_moves(c1, c2):
        moves = []
        for i in range(len(c1)):
            for j in range(i + 1, len(c1)):
                new_c1, delta = swap_in_cycle_edges(c1[:], i, j, distance_matrix)
                if delta < 0:
                    moves.append((delta, ("intra", i, j, "c1")))
        for i in range(len(c2)):
            for j in range(i + 1, len(c2)):
                new_c2, delta = swap_in_cycle_edges(c2[:], i, j, distance_matrix)
                if delta < 0:
                    moves.append((delta, ("intra", i, j, "c2")))
        for i in range(len(c1)):
            for j in range(len(c2)):
                _, _, delta1, delta2 = swap_between_cycles(c1[:], c2[:], i, j, distance_matrix)
                if delta1 + delta2 < 0:
                    moves.append((delta1 + delta2, ("inter", i, j)))
        return sorted(moves, key=lambda x: x[0])

    LM = deque(evaluate_moves(cycle1, cycle2))

    while LM:
        applied = False
        for _ in range(len(LM)):
            delta, move = LM.popleft()
            if move[0] == "intra":
                _, i, j, which = move
                if which == "c1":
                    new_cycle, delta_check = swap_in_cycle_edges(cycle1[:], i, j, distance_matrix)
                    if delta_check == delta:
                        cycle1 = new_cycle
                        applied = True
                        break
                else:
                    new_cycle, delta_check = swap_in_cycle_edges(cycle2[:], i, j, distance_matrix)
                    if delta_check == delta:
                        cycle2 = new_cycle
                        applied = True
                        break
            else:  # inter
                _, i, j = move
                new_c1, new_c2, delta1, delta2 = swap_between_cycles(cycle1[:], cycle2[:], i, j, distance_matrix)
                if delta1 + delta2 == delta:
                    cycle1, cycle2 = new_c1, new_c2
                    applied = True
                    break
        if applied:
            LM = deque(evaluate_moves(cycle1, cycle2))  # regenerate LM
        else:
            break

    return cycle1, cycle2


def process_file(file_paths, init_methods, modes, algorithms, num_iterations=1):
    for file_path in file_paths:
        results = defaultdict(lambda: {
        "result_list": [],
        "best_result": float("inf"),
        "best_cycle1": None,
        "best_cycle2": None
        })
        best_solutions = [] 
        print(f"Processing: {file_path}")
        distance_matrix, coordinates, n = read_instance(file_path)
        for init_method in init_methods:
            for mode in modes:
                for algorithm in algorithms:
                    config_key = f"{init_method}-{mode}-{algorithm}"

                    for i in range(num_iterations):
                        print(config_key, i)
                        cycle1, cycle2 = random_cycles(n)
                        start_time = time.time()

                        if algorithm == "steepest":
                            cycle1, cycle2 = local_search(cycle1, cycle2, distance_matrix,coordinates, mode,algorithm)
                        elif algorithm == "lm":
                            cycle1, cycle2 = lm_local_search(cycle1, cycle2, distance_matrix)

                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        print(f"Elapsed time: {elapsed_time:.2f} seconds")

                        length = cycle_length(cycle1, distance_matrix) + cycle_length(cycle2, distance_matrix)
                        results[config_key]["result_list"].append(length)
                        print(length)

                        # Check if it's the best result
                        if length < results[config_key]["best_result"]:
                            results[config_key]["best_result"] = length
                            results[config_key]["best_cycle1"] = cycle1
                            results[config_key]["best_cycle2"] = cycle2
                            results[config_key]["name"] = f"{init_method}-{mode}-{algorithm}"

                    # Add best solution for plotting
                    best_cycle1 = results[config_key]["best_cycle1"]
                    best_cycle2 = results[config_key]["best_cycle2"]
                    name = results[config_key]["name"]
                    best_solutions.append((best_cycle1, best_cycle2, coordinates,name))

        f = os.path.basename(file_path)
        f = f.split('.')[0]
        plot_multiple_solutions(best_solutions,f)

        save_best_results(results,f)

    return results

if __name__ == "__main__":
    folder_path = "../data/"
    init_methods = ["rand"]
    modes = ["edges"]
    algorithms = ["lm","steepest"]
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))
    num_iterations = 2

    results = process_file(file_paths, init_methods, modes, algorithms, num_iterations)