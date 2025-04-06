import sys
import os
import matplotlib.pyplot as plt
import random
import glob
import time
from collections import defaultdict
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../lab1")))
from lab1 import (
    read_instance,
    heuristic_algorithm_regret_weighted,
    plot_solution,
    cycle_length,
)

folder_path = "../data/"


def plot_multiple_solutions(cycles_and_coordinates,file):
    plt.figure(figsize=(15, 15))  # Size of the entire figure (width, height)

    for i, (cycle1, cycle2, coordinates,name) in enumerate(cycles_and_coordinates, start=1):
        print(i,cycle1)
        plt.subplot(3, 4, i)  # Change to 3 rows and 4 columns for 12 plots
        
        plot_solution(cycle1, cycle2, coordinates, None)  # Plot each solution
        
        plt.title(name)  # Title for each plot
        
    plt.tight_layout()  # Adjust the layout for better spacing
    #plt.show()
    plt.savefig(f"plt_{file}", bbox_inches='tight')  # Zapis do pliku
    plt.close('all')

# Function to save results to a text file
def save_best_results(results,file):
    """
    Funkcja zapisuje najlepsze wyniki w formacie:
    Instancja 1 Instancja 2 ...
    Metoda 1    średnia (min – max)  ...
    Metoda 2    średnia (min – max)  ...

    :param results: słownik, gdzie klucze to config_key, a wartości to słowniki z wynikami.
    """
    # Zakładamy, że liczba instancji to liczba wyników w result_list
    num_instances = len(next(iter(results.values()))['result_list'])
    instance_headers = [f'Instancja {i+1}' for i in range(num_instances)]
    header = ['Metoda'] + instance_headers

    output = []

    for config_key, config_data in results.items():
        row = [config_key]
        for instance_result in config_data['result_list']:
            mean_value = instance_result  # w tym przypadku to jedna wartość, więc mean = min = max
            row.append(f'{mean_value:.2f} ({mean_value:.0f} – {mean_value:.0f})')
        output.append(row)

    # Tworzymy DataFrame i wypisujemy lub zapisujemy
    df = pd.DataFrame(output, columns=header)
    print(df.to_string(index=False))
    df.to_csv(f"results_{file}", index=False, sep=';')
    return df



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
# Funkcja dla algorytmu losowego błądzenia
def measure_time_for_local_search(algorithm, cycle1, cycle2, distance_matrix, mode,coordinates):
    start_time = time.time()
    cycle1, cycle2 = local_search(cycle1, cycle2, distance_matrix,coordinates, mode,algorithm)
    end_time = time.time()
    
    return end_time - start_time, cycle1, cycle2
def random_walk(cycle1, cycle2, distance_matrix, max_time):
    max_time =0.1
    start_time = time.time()
    while time.time() - start_time < max_time:
        # Losowy ruch - wybieramy losowy indeks w cyklach i zamieniamy
        i = random.randint(0, len(cycle1) - 1)
        j = random.randint(0, len(cycle2) - 1)
        # Zamiana pomiędzy cyklami (lub w obrębie cykli w zależności od logiki)
        cycle1, cycle2, delta1, delta2 = swap_between_cycles(cycle1, cycle2, i, j, distance_matrix)
    # Zwracamy najlepsze znalezione rozwiązanie (najlepszy cykl)
    return cycle1, cycle2


def swap_between_cycles(cycle1, cycle2, idx1, idx2, distance_matrix):
    # Swap two nodes between cycles
    length = cycle_length(cycle1,distance_matrix) + cycle_length(cycle2,distance_matrix)
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
    if length < cycle_length(cycle1,distance_matrix) + cycle_length(cycle2,distance_matrix):
        return cycle1, cycle2, 1, 1

    return cycle1, cycle2, delta1, delta2


def swap_in_cycle_nodes(cycle, idx1, idx2, distance_matrix):
    # Swap two nodes in the same cycle
    length = cycle_length(cycle,distance_matrix)
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
    if length < cycle_length(cycle,distance_matrix):
        return cycle, 1

    return cycle, delta

# Funkcja Greedy Local Search (w wersji zachłannej)
def greedy_local_search(cycle1, cycle2, distance_matrix, mode ="edges"):
    indices1 = list(range(len(cycle1)))
    indices2 = list(range(len(cycle2)))
    random.shuffle(indices1)
    random.shuffle(indices2)

    for cycle, other_cycle, idx_cycle, idx_other in [
        (cycle1, cycle2, indices1, indices2),
        (cycle2, cycle1, indices2, indices1)
    ]:
        for i in idx_cycle:
            for j in idx_cycle:
                if i == j:
                    continue
                if mode == "edges":
                    new_cycle, delta = swap_in_cycle_edges(cycle, i, j, distance_matrix)
                else:
                     new_cycle, delta = swap_in_cycle_nodes(cycle[:], i, j, distance_matrix)
                if delta < 0:
                    return new_cycle, other_cycle
                else:
                    new_cycle, new_other_cycle, delta1, delta2 = swap_between_cycles(cycle[:], other_cycle[:], i, j, distance_matrix)
                    if delta1 + delta2 < 0:
                        return new_cycle, new_other_cycle

    return cycle1, cycle2


def steepest_local_search(cycle1, cycle2, distance_matrix,mode="cycles"):
    best_delta = float('inf') 
    best_swap = None
    for cycle, other_cycle in [(cycle1, cycle2), (cycle2, cycle1)]:
        for i in range(len(cycle)):
            for j in range(len(other_cycle)):
                if i == j:
                    continue
                if mode == "edges":
                    new_cycle, delta = swap_in_cycle_edges(cycle, i, j, distance_matrix)
                else:
                     new_cycle, delta = swap_in_cycle_nodes(cycle[:], i, j, distance_matrix)
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (new_cycle, other_cycle)
                #else:
                 #   new_cycle, new_other_cycle, delta1, delta2 = swap_between_cycles(cycle[:], other_cycle[:], i, j, distance_matrix)
                 #   if delta1 + delta2 < best_delta:
                 #       best_delta = delta1 + delta2
                  #      best_swap = (new_cycle, new_other_cycle)
    return best_swap[0],best_swap[1]

def local_search(cycle1, cycle2, distance_matrix,coordinates,mode="cycles",type="greedy"):
    it =0
    length = cycle_length(cycle1,distance_matrix) + cycle_length(cycle2,distance_matrix)
    while True:
        oc1 = cycle1.copy()
        oc2 =cycle2.copy()
        if type == "greedy":
            cycle1, cycle2 = greedy_local_search(cycle1, cycle2, distance_matrix,mode)
        else: 
            cycle1, cycle2 = steepest_local_search(cycle1, cycle2, distance_matrix,mode)
        new_length = cycle_length(cycle1,distance_matrix) + cycle_length(cycle2,distance_matrix)
        print(it,length,new_length)
        it+=1
        if length == new_length:
            break
        elif length < new_length:
            break
        length = new_length

    return cycle1, cycle2

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
                        if init_method == "rand":
                            cycle1, cycle2 = random_cycles(n)
                        else:
                            cycle1, cycle2 = heuristic_algorithm_regret_weighted(distance_matrix, n)

                        if algorithm != "random_walk":
                            time_taken, cycle1, cycle2 = measure_time_for_local_search(algorithm, cycle1, cycle2, distance_matrix, mode, coordinates)
                        else:
                            cycle1, cycle2 = random_walk(cycle1, cycle2, distance_matrix, time_taken)

                        # Calculate the length of the solution
                        length = cycle_length(cycle1, distance_matrix) + cycle_length(cycle2, distance_matrix)
                        results[config_key]["result_list"].append(length)

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

        # After completing the calculations for all files, plot the best solutions
        f = os.path.basename(file_path)
        f = f.split('.')[0]
        plot_multiple_solutions(best_solutions,f)

        # Save the results to a text file
        save_best_results(results,f)

    return results

if __name__ == "__main__":
    init_methods = ["rand", "reg"]
    modes = ["edges","nodes"]
    algorithms = ["greedy", "steepest","random_walk"]
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))
    num_iterations = 1

    results = process_file(file_paths, init_methods, modes, algorithms, num_iterations)
