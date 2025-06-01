import glob
from statistics import mean
import sys, os, re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
TIME_TO_CAL = 188.37
POP_SIZE = 20
K = 5

for name in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, name)
    if (
        re.match(r"lab\d+", name)
        and os.path.isdir(full_path)
        and full_path != current_dir
    ):
        sys.path.append(full_path)

from lab1 import read_instance, cycle_length
from lab2 import random_cycles, greedy_local_search, local_search
from lab5 import HAE


def both_in_cycle(node1, node2, cycle1, cycle2):
    if node1 in cycle1 and node2 in cycle1:
        return True
    if node1 in cycle2 and node2 in cycle2:
        return True
    return False


def calc_similarity_nodes(solution1, solution2, n):
    solution1 = (set(solution1[0]), set(solution1[1]))
    solution2 = (set(solution2[0]), set(solution2[1]))

    similarity = 0
    for node1 in range(n):
        for node2 in range(node1 + 1, n, 1):
            if both_in_cycle(
                node1, node2, solution1[0], solution1[1]
            ) and both_in_cycle(node1, node2, solution2[0], solution2[1]):
                similarity += 1

    return similarity


def calc_similarity_edges(solution1, solution2):

    edges1 = set()
    edges2 = set()

    for cycle in solution1:
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            edges1.add((min(a, b), max(a, b)))

    for cycle in solution2:
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            edges2.add((min(a, b), max(a, b)))

    similarity = len(edges1.intersection(edges2))
    return similarity


def plot_similarities(
    random_solutions,
    good_solution,
    n,
    distance_matrix,
    mode,
    data_name,
):

    similarities_good = []
    similarities_random_mean = []
    solution_values = []

    for random_solution in random_solutions:

        solution_values.append(
            cycle_length(random_solution[0], distance_matrix)
            + cycle_length(random_solution[1], distance_matrix)
        )

        print(solution_values[-1])

        if mode == "nodes":
            similarities_good.append(
                calc_similarity_nodes(random_solution, good_solution, n)
            )
            similarities_random_mean.append(
                mean(
                    [
                        calc_similarity_nodes(random_solution, other, n)
                        for other in random_solutions
                        if other != random_solution
                    ]
                )
            )
        elif mode == "edges":
            similarities_good.append(
                calc_similarity_edges(random_solution, good_solution)
            )
            similarities_random_mean.append(
                mean(
                    [
                        calc_similarity_edges(random_solution, other)
                        for other in random_solutions
                        if other != random_solution
                    ]
                )
            )

    # Calculate correlations
    if len(solution_values) > 1:
        corr_good, _ = pearsonr(solution_values, similarities_good)
        corr_random, _ = pearsonr(solution_values, similarities_random_mean)
    else:
        corr_good = float("nan")
        corr_random = float("nan")

    # Plot similarities to good solution
    plt.figure(figsize=(12, 5))
    plt.scatter(solution_values, similarities_good, color="blue", alpha=0.7)
    plt.xlabel("Wartość funkcji celu")
    plt.ylabel("Podobieństwo")
    plt.title(
        f"Podobieństwo do dobrego rozwiązania\n"
        f"Miara podobieństwa: {'wierzchołkowa' if mode == 'nodes' else 'krawędziowa'}\n"
        f"Zbiór danych: {data_name}\n"
        f"Korelacja: {corr_good:.3f}"
    )
    plt.tight_layout()
    plt.savefig(f"similarity-good_{mode}_{data_name}.png")
    plt.clf()

    # Plot mean similarities to other random solutions
    plt.scatter(solution_values, similarities_random_mean, color="green", alpha=0.7)
    plt.xlabel("Wartość funkcji celu")
    plt.ylabel("Podobieństwo")
    plt.title(
        f"Średnie podobieństwo do pozostałych rozwiązań losowych\n"
        f"Miara podobieństwa: {'wierzchołkowa' if mode == 'nodes' else 'krawędziowa'}\n"
        f"Zbiór danych: {data_name}\n"
        f"Korelacja: {corr_random:.3f}"
    )
    plt.tight_layout()
    plt.savefig(f"similarity-random-mean_{mode}_{data_name}.png")
    plt.clf()


def process_file(file_paths):
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        print(f"Constants: TIME_TO_CAL={TIME_TO_CAL}, POP_SIZE={POP_SIZE}, K={K}")
        distance_matrix, coordinates, n = read_instance(file_path)

        # Initialize random optimized solutions
        random_solutions = []
        for _ in range(1000):
            cycle1, cycle2 = random_cycles(n)
            # cycle1, cycle2 = greedy_local_search(
            #     cycle1, cycle2, distance_matrix, "edges"
            # )
            cycle1, cycle2 = local_search(
                cycle1, cycle2, distance_matrix, coordinates, "edges", "greedy"
            )
            random_solutions.append((cycle1, cycle2))

        # Find good solutions using HAE
        good_cycle1, good_cycle2, _ = HAE(
            distance_matrix, n, with_local_search=True, mut=True
        )
        good_solution = (good_cycle1, good_cycle2)
        print(
            "Good solution found using HAE:",
            cycle_length(good_solution[0], distance_matrix)
            + cycle_length(good_solution[1], distance_matrix),
        )

        # plot_similarities(
        #     random_solutions,
        #     good_solution,
        #     n,
        #     distance_matrix,
        #     mode="nodes",
        #     data_name=os.path.basename(file_path),
        # )

        plot_similarities(
            random_solutions,
            good_solution,
            n,
            distance_matrix,
            mode="edges",
            data_name=os.path.basename(file_path),
        )


if __name__ == "__main__":
    folder_path = "../data/"
    file_paths = glob.glob(os.path.join(folder_path, "*.tsp"))
    results = process_file(file_paths)
