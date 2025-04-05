import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../lab1")))
from lab1 import read_instance, heuristic_algorithm_regret_weighted, plot_solution, cycle_length

file_paths = ["./lab1/kroA100.tsp"]


def simple_plot(cycle1, cycle2, coordinates):
    plt.figure(figsize=(10, 10))
    plot_solution(cycle1, cycle2, coordinates, None)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    for file_path in file_paths:
        distance_matrix, coordinates, n = read_instance(file_path)

        cycle1, cycle2 = heuristic_algorithm_regret_weighted(distance_matrix, n)
        simple_plot(cycle1, cycle2, coordinates)
        