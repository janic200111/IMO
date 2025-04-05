import sys
import os
import matplotlib.pyplot as plt
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../lab1")))
from lab1 import (
    read_instance,
    heuristic_algorithm_regret_weighted,
    plot_solution,
    cycle_length,
)

file_paths = ["./lab1/kroA100.tsp"]


def simple_plot(cycle1, cycle2, coordinates):
    plt.figure(figsize=(10, 10))
    plot_solution(cycle1, cycle2, coordinates, None)
    plt.legend()
    plt.show()


def random_cycles(distance_matrix, n):
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

    for file_path in file_paths:
        distance_matrix, coordinates, n = read_instance(file_path)

        # Crop instance
        # nice for testing functions
        n = 12
        distance_matrix = distance_matrix[:n, :n]
        coordinates = coordinates[:n]
        cycle1 = [11, 2, 4, 1, 3, 5, 9, 10]
        cycle2 = [0]
        simple_plot(cycle1, cycle2, coordinates)
        print("Cycle1 length:", cycle_length(cycle1, distance_matrix))
        # cycle1, delta = swap_in_cycle_edges(cycle1, 3, 7, distance_matrix)
        cycle1, delta = swap_in_cycle_nodes(cycle1, 3, 7, distance_matrix)
        print("Cycle1 length:", cycle_length(cycle1, distance_matrix))
        print("Delta:", delta)
        simple_plot(cycle1, cycle2, coordinates)
        # end nice for testing functions

        exit()
        for idx1 in range(5):
            idx1 = random.randint(0, len(cycle1) - 1)
            idx2 = random.randint(0, len(cycle1) - 1)
            print(f"Swap {cycle1[idx1]} with {cycle1[idx2]}")
            # cycle1, cycle2 = swap_between_cycles(cycle1, cycle2, idx1, idx2)
            cycle1 = swap_in_cycle_nodes(cycle1, idx1, idx2)
            simple_plot(cycle1, cycle2, coordinates)
