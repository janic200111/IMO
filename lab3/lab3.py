import glob
import time
from collections import defaultdict, deque
import sys, os, re
import pandas as pd

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
    heuristic_algorithm_regret_weighted,
    cycle_length,
)

from lab2 import (
    swap_in_cycle_nodes,
    swap_in_cycle_edges,
    swap_between_cycles,
    local_search,
    random_cycles,
    plot_multiple_solutions,
)


def lm_local_search(cycle1, cycle2, distance_matrix):
    def evaluate_moves(c1, c2):
        # Funkcja generuje wszystkie możliwe ruchy poprawiające rozwiązanie
        moves = []
        # Intra-cycle moves dla pierwszego cyklu (wymiana w ramach jednego cyklu)
        for i in range(len(c1)):
            for j in range(i + 1, len(c1)):
                new_c1, delta = swap_in_cycle_edges(c1[:], i, j, distance_matrix)
                if delta < 0:
                    moves.append((delta, ("intra", i, j, "c1")))
        # Intra-cycle moves dla drugiego cyklu
        for i in range(len(c2)):
            for j in range(i + 1, len(c2)):
                new_c2, delta = swap_in_cycle_edges(c2[:], i, j, distance_matrix)
                if delta < 0:
                    moves.append((delta, ("intra", i, j, "c2")))
        # Inter-cycle moves (wymiana wierzchołków między cyklami)
        for i in range(len(c1)):
            for j in range(len(c2)):
                _, _, delta1, delta2 = swap_between_cycles(
                    c1[:], c2[:], i, j, distance_matrix
                )
                if delta1 + delta2 < 0:
                    moves.append((delta1 + delta2, ("inter", i, j)))
        # Zwracamy wszystkie ruchy posortowane według poprawy (najlepsze pierwsze)
        return sorted(moves, key=lambda x: x[0])

    LM = deque(evaluate_moves(cycle1, cycle2))

    while LM:
        applied = False
        idx = 0
        LM_list = list(LM)
        while idx < len(LM_list):
            delta, move = LM_list[idx]
            if move[0] == "intra":
                # Ruch wewnątrz jednego cyklu
                _, i, j, which = move
                if which == "c1":
                    new_cycle, delta_check = swap_in_cycle_edges(
                        cycle1[:], i, j, distance_matrix
                    )
                    if delta_check == delta:
                        # Jeśli zmiana dalej poprawia tak samo, stosujemy ruch
                        cycle1 = new_cycle
                        applied = True
                        break
                    else:
                        # Ruch nieaktualny - przechodzimy do kolejnego
                        idx += 1
                else:
                    new_cycle, delta_check = swap_in_cycle_edges(
                        cycle2[:], i, j, distance_matrix
                    )
                    if delta_check == delta:
                        cycle2 = new_cycle
                        applied = True
                        break
                    else:
                        idx += 1
            else:
                # Ruch między cyklami
                _, i, j = move
                new_c1, new_c2, delta1, delta2 = swap_between_cycles(
                    cycle1[:], cycle2[:], i, j, distance_matrix
                )
                if delta1 + delta2 == delta:
                    # Jeśli zmiana nadal poprawna, wykonujemy
                    cycle1, cycle2 = new_c1, new_c2
                    applied = True
                    break
                else:
                    # Ruch nieaktualny - przechodzimy dalej
                    idx += 1
        if applied:
            # Po każdej udanej zmianie, generujemy na nowo wszystkie możliwe ruchy
            LM = deque(evaluate_moves(cycle1, cycle2))
        else:
            # Jeśli żaden ruch nie został wykonany, kończymy przeszukiwanie
            break

    return cycle1, cycle2


def candidate_moves(cycle1, cycle2, distance_matrix):

    def find_closest_neighbours(distance_matrix):
        n = len(distance_matrix)
        closest_neighbours = {}
        for i in range(n):
            closest_neighbours[i] = sorted(
                range(n), key=lambda x: distance_matrix[i][x]
            )[1:11]
            # 10 closest neighbours excluding itself

        return closest_neighbours

    closest_neighbours = find_closest_neighbours(distance_matrix)
    n = len(distance_matrix)

    is_cycle1 = [1 if i in set(cycle1) else 0 for i in range(n)]

    while True:

        best_delta = 0
        best_cycles = None

        for i in range(n):
            for j in closest_neighbours[i]:

                if is_cycle1[i] == is_cycle1[j]:  # both in the same cycle
                    if is_cycle1[i] == 1:
                        new_cycle_1, delta = swap_in_cycle_edges(
                            cycle1[:], cycle1.index(i), cycle1.index(j), distance_matrix
                        )
                        new_cycles = (new_cycle_1, cycle2)
                    elif is_cycle1[i] == 0:
                        new_cycle_2, delta = swap_in_cycle_edges(
                            cycle2[:], cycle2.index(i), cycle2.index(j), distance_matrix
                        )
                        new_cycles = (cycle1, new_cycle_2)

                elif is_cycle1[i] != is_cycle1[j]:  # swap between cycles
                    if is_cycle1[i] == 1:
                        idx_cycle1 = cycle1.index(i)
                        idx_cycle2 = cycle2.index(j)
                    else:
                        idx_cycle1 = cycle2.index(i)
                        idx_cycle2 = cycle1.index(j)

                    new_cycle_1, new_cycle_2, delta1, delta2 = swap_between_cycles(
                        cycle1[:],
                        cycle2[:],
                        idx_cycle1,
                        ((idx_cycle2 + 1) % len(cycle2)),
                        distance_matrix,
                    )
                    delta = delta1 + delta2
                    new_cycles = (new_cycle_1, new_cycle_2)

                if delta < best_delta:
                    best_delta = delta
                    best_cycles = new_cycles

        if best_delta < 0:
            cycle1, cycle2 = best_cycles
            is_cycle1 = [1 if i in set(cycle1) else 0 for i in range(n)]
        else:
            break

    return cycle1, cycle2


def process_file(file_paths, init_methods, modes, algorithms, num_iterations=1):
    for file_path in file_paths:
        results = defaultdict(
            lambda: {
                "result_list": [],
                "best_result": float("inf"),
                "best_cycle1": None,
                "best_cycle2": None,
            }
        )
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
                            cycle1, cycle2 = local_search(
                                cycle1,
                                cycle2,
                                distance_matrix,
                                coordinates,
                                mode,
                                algorithm,
                            )
                        elif algorithm == "lm":
                            cycle1, cycle2 = lm_local_search(
                                cycle1, cycle2, distance_matrix
                            )
                        elif algorithm == "candidate":
                            cycle1, cycle2 = candidate_moves(
                                cycle1, cycle2, distance_matrix
                            )
                        elif algorithm == "regret":
                            cycle1, cycle2 = heuristic_algorithm_regret_weighted(
                                distance_matrix, len(distance_matrix)
                            )

                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        print(f"Elapsed time: {elapsed_time:.2f} seconds")

                        length = cycle_length(cycle1, distance_matrix) + cycle_length(
                            cycle2, distance_matrix
                        )
                        results[config_key]["result_list"].append(length)
                        print(length)

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
        row = [
            config_key,
            avg_result,
            min_result,
            max_result,
        ]
        output.append(row)

    df = pd.DataFrame(output, columns=["Configuration", "Avg", "Min", "Max"])
    df.to_csv(f"results_{file_name}.csv", index=False)


if __name__ == "__main__":
    folder_path = "../data/"
    init_methods = ["rand"]
    modes = ["edges"]
    algorithms = ["lm","steepest", "candidate", "regret"]
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))
    num_iterations = 2

    results = process_file(file_paths, init_methods, modes, algorithms, num_iterations)
