import sys
import os
import matplotlib.pyplot as plt
import random
import glob
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../lab1")))
from lab1 import (
    read_instance,
    heuristic_algorithm_regret_weighted,
    plot_solution,
    cycle_length,
)

folder_path = "../data/"



def simple_plot(cycle1, cycle2, coordinates):
    plt.figure(figsize=(10, 10))
    plot_solution(cycle1, cycle2, coordinates, None)
    plt.legend()
    plt.show()


def random_cycles(n):
    indexes = list(range(n))
    random.shuffle(indexes)
    cycle1 = indexes[: n // 2]
    cycle2 = indexes[n // 2 :]
    return cycle1, cycle2


###
# DELTA = NEW - OLD  <--- minimizing
###


def swap_between_cycles(cycle1, cycle2, idx1, idx2, distance_matrix):
    # Swap two nodes between cycles

    l1 = len(cycle1)
    delta1 = (
        0
        + distance_matrix[cycle1[((idx1 - 1) + l1) % l1]][cycle2[idx2]]
        + distance_matrix[cycle1[(idx1 + 1) % l1]][cycle2[idx2]]
        - distance_matrix[cycle1[((idx1 - 1) + l1) % l1]][cycle1[idx1]]
        - distance_matrix[cycle1[idx1]][cycle1[(idx1 + 1) % l1]]
    )

    l2 = len(cycle2)
    delta2 = (
        0
        + distance_matrix[cycle2[((idx2 - 1) + l2) % l2]][cycle1[idx1]]
        + distance_matrix[cycle2[(idx2 + 1) % l2]][cycle1[idx1]]
        - distance_matrix[cycle2[((idx2 - 1) + l2) % l2]][cycle2[idx2]]
        - distance_matrix[cycle2[idx2]][cycle2[(idx2 + 1) % l2]]
    )

    cycle1[idx1], cycle2[idx2] = cycle2[idx2], cycle1[idx1]

    return cycle1, cycle2, delta1, delta2


def swap_in_cycle_nodes(cycle, idx1, idx2, distance_matrix):
    # Swap two nodes in the same cycle

    l = len(cycle)
    delta = (
        0
        + distance_matrix[cycle[((idx1 - 1) + l) % l]][cycle[idx2]]
        + distance_matrix[cycle[(idx1 + 1) % l]][cycle[idx2]]
        + distance_matrix[cycle[((idx2 - 1) + l) % l]][cycle[idx1]]
        + distance_matrix[cycle[(idx2 + 1) % l]][cycle[idx1]]
        - distance_matrix[cycle[((idx1 - 1) + l) % l]][cycle[idx1]]
        - distance_matrix[cycle[idx1]][cycle[(idx1 + 1) % l]]
        - distance_matrix[cycle[((idx2 - 1) + l) % l]][cycle[idx2]]
        - distance_matrix[cycle[idx2]][cycle[(idx2 + 1) % l]]
    )

    cycle[idx1], cycle[idx2] = cycle[idx2], cycle[idx1]

    return cycle, delta

# Funkcja Greedy Local Search (w wersji zachłannej)
def greedy_local_search(cycle1, cycle2, distance_matrix, mode ="cycles"):
    indices1 = list(range(len(cycle1)))
    indices2 = list(range(len(cycle2)))
    random.shuffle(indices1)
    random.shuffle(indices2)

    for cycle, other_cycle, idx_cycle, idx_other in [
        (cycle1, cycle2, indices1, indices2),
        (cycle2, cycle1, indices2, indices1)
    ]:
        for i in idx_cycle:
            if mode == "cycles":
                for j in idx_other:
                    new_cycle, new_other_cycle, delta1, delta2 = swap_between_cycles(cycle, other_cycle, i, j, distance_matrix)
                    if delta1 + delta2 < 0:
                        return new_cycle, new_other_cycle
            else:
                for j in idx_cycle:
                    if i == j:
                        continue
                    new_cycle, delta = swap_in_cycle_edges(cycle, i, j, distance_matrix)
                    if delta < 0:
                        if cycle == cycle1:
                            return new_cycle, cycle2
                        else:
                            return cycle1, new_cycle

    return cycle1, cycle2


def steepest_local_search(cycle1, cycle2, distance_matrix,mode="cycles"):
    best_delta = float('inf') 
    best_swap = None

    for cycle, other_cycle in [(cycle1, cycle2), (cycle2, cycle1)]:
        for i in range(len(cycle)):
            if mode == "cycles":
                for j in range(len(other_cycle)):
                    new_cycle, new_other_cycle, delta1, delta2 = swap_between_cycles(cycle, other_cycle, i, j, distance_matrix)
                    if delta1 + delta2 < best_delta:
                        best_delta = delta1 + delta2
                        best_swap = (new_cycle, new_other_cycle)
            else:
                for j in range(len(cycle)):
                    if i == j:
                        continue
                    new_cycle, delta = swap_in_cycle_edges(cycle, i, j, distance_matrix)
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (new_cycle, other_cycle)
    return best_swap[0],best_swap[1]





def swap_in_cycle_edges(cycle, idx1, idx2, distance_matrix):
    # Swap two edges in the same cycle

    if idx1 > idx2:
        idx1, idx2 = idx2, idx1  # the cycles are not directed

    l = len(cycle)
    delta = (
        0
        - distance_matrix[cycle[idx1]][cycle[(idx1 + 1) % l]]
        - distance_matrix[cycle[idx2]][cycle[(idx2 + 1) % l]]
        + distance_matrix[cycle[idx1]][cycle[idx2]]
        + distance_matrix[cycle[(idx1 + 1) % l]][cycle[(idx2 + 1) % l]]
    )

    cycle = (
        cycle[: idx1 + 1]
        + list(reversed(cycle[idx1 + 1 : idx2 + 1]))
        + cycle[idx2 + 1 :]
    )

    return cycle, delta


if __name__ == "__main__":
    results = {}
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))
    NUM_ITTER = 1  # Zmienna do iteracji

    for file_path in file_paths:
        print(f"Processing: {file_path}")
        init = ["rand", "reg"]
        modes = ["cycles", "edges"]
        alg = ["greedy", "steepset"]
        distance_matrix, coordinates, n = read_instance(file_path)

        results[file_path] = defaultdict(lambda: {
            "result_list": [],
            "best_result": float("inf"),
            "best_cycle1": None,
            "best_cycle2": None
        })

        for i in init:
            for m in modes:
                for a in alg:
                    config_key = f"{i}-{m}-{a}"

                    for it in range(NUM_ITTER):
                        print(f"{config_key}, iteration {it}")
                        if i == "rand":
                            cycle1, cycle2 = random_cycles(n)
                        else:
                            cycle1, cycle2 = heuristic_algorithm_regret_weighted(distance_matrix, n)

                        if a == "greedy":
                            cycle1, cycle2 = greedy_local_search(cycle1, cycle2, distance_matrix, m)
                        else:
                            cycle1, cycle2 = steepest_local_search(cycle1, cycle2, distance_matrix, m)

                        length = cycle_length(cycle1, distance_matrix) + cycle_length(cycle2, distance_matrix)
                        results[file_path][config_key]["result_list"].append(length)

                        if length < results[file_path][config_key]["best_result"]:
                            results[file_path][config_key]["best_result"] = length
                            results[file_path][config_key]["best_cycle1"] = cycle1
                            results[file_path][config_key]["best_cycle2"] = cycle2

    # Budowanie tabeli wyników
    instances = list(results.keys())
    header = f"{'Metoda':<35}" + "".join([f"{inst:<30}" for inst in instances])
    table_lines = [header, "-" * len(header)]

    all_methods = set()
    for file_data in results.values():
        all_methods.update(file_data.keys())
    all_methods = sorted(all_methods)

    for method in all_methods:
        row = f"{method:<35}"
        for instance in instances:
            if method in results[instance]:
                data = results[instance][method]
                rlist = data["result_list"]
                avg = round(sum(rlist) / len(rlist), 2) if rlist else "N/A"
                min_val = min(rlist) if rlist else "N/A"
                max_val = max(rlist) if rlist else "N/A"
                row += f"{avg} ({min_val} – {max_val})".ljust(30)
            else:
                row += "Brak danych".ljust(30)
        table_lines.append(row)
    # Zapis do pliku
    output_file = "wyniki_tabela_full.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))

    # Wyświetlenie tabeli
    print("\n".join(table_lines))
    print(f"\nWyniki zapisane do pliku: {output_file}")
    """
    # Rysowanie najlepszych rozwiązań
    for file, methods_results in results.items():
        print(f"\n=== Wyniki dla pliku: {file} ===")
        distance_matrix, coordinates, n = read_instance(file)
        plt.figure(figsize=(10, 10))
        for i, (method, data) in enumerate(methods_results.items()):
            print(f"\nMetoda: {method}")
            print(f" - Najlepszy wynik: {data['min_result']}")
            print(f" - Najgorszy wynik: {data['max_result']}")
            print(f" - Średnia wyników: {data['sum_results']/NUM_ITTER}")
            plot_solution(data["best_cycle1"], data["best_cycle2"], coordinates, i + 1)
            plt.title(method)
        plt.suptitle(file)
        plt.legend()
        plt.show()


        # Crop instance
        # nice for testing functions
        #n = 10
        #distance_matrix = distance_matrix[:n, :n]
        #coordinates = coordinates[:n]
        #cycle1 = [11, 2, 4, 1, 3, 5, 9, 10]
        #cycle2 = [0]
        cycle1, cycle2 = random_cycles(n)
        #cycle1, cycle2 = greedy_local_search(cycle1,cycle2,distance_matrix)
        #simple_plot(cycle1, cycle2, coordinates)
        #cycle1, cycle2 = greedy_local_search(cycle1,cycle2,distance_matrix,"edges")
        #simple_plot(cycle1, cycle2, coordinates)
        #simple_plot(cycle1, cycle2, coordinates)
        cycle1, cycle2 = steepest_local_search(cycle1,cycle2,distance_matrix)
        simple_plot(cycle1, cycle2, coordinates)
        cycle1, cycle2 = steepest_local_search(cycle1,cycle2,distance_matrix,"edges")
        simple_plot(cycle1, cycle2, coordinates)
        print("Cycle1 length:", cycle_length(cycle1, distance_matrix))
        # cycle1, delta = swap_in_cycle_edges(cycle1, 3, 7, distance_matrix)
        #cycle1, delta = swap_in_cycle_nodes(cycle1, 3, 7, distance_matrix)
        #print("Cycle1 length:", cycle_length(cycle1, distance_matrix))
        #print("Delta:", delta)
        #simple_plot(cycle1, cycle2, coordinates)
        # end nice for testing functions

        exit()
        for idx1 in range(5):
            idx1 = random.randint(0, len(cycle1) - 1)
            idx2 = random.randint(0, len(cycle1) - 1)
            print(f"Swap {cycle1[idx1]} with {cycle1[idx2]}")
            # cycle1, cycle2 = swap_between_cycles(cycle1, cycle2, idx1, idx2)
            cycle1 = swap_in_cycle_nodes(cycle1, idx1, idx2)
            simple_plot(cycle1, cycle2, coordinates)"
        """
