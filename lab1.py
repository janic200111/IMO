import math
import random
import numpy as np
import matplotlib.pyplot as plt


def read_instance(file_path):
    coordinates = []
    reading_coordinates = False

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                reading_coordinates = True
                continue
            if line.startswith("EOF"):
                break

            if reading_coordinates:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coordinates.append((x, y))
                    except ValueError:
                        print(f"Błąd podczas konwersji: {line}")

    n = len(coordinates)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            dist = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist

    return distance_matrix, coordinates, n


def greedy_algorithm_nearest_neighbour(distance_matrix, n):
    start1 = random.randint(0, n - 1)
    distances_from_start = [distance_matrix[start1][i] for i in range(n)]
    farthest_node = distances_from_start.index(max(distances_from_start))

    cycle1 = [start1]
    visited = set(cycle1)

    cycle2 = [farthest_node]
    visited.add(farthest_node)

    while len(cycle1) + len(cycle2) < n:
        last_node1 = cycle1[-1]
        nearest_node1 = min(
            (i for i in range(n) if i not in visited),
            key=lambda x: distance_matrix[last_node1][x],
        )
        cycle1.append(nearest_node1)
        visited.add(nearest_node1)

        last_node2 = cycle2[-1]
        nearest_node2 = min(
            (i for i in range(n) if i not in visited),
            key=lambda x: distance_matrix[last_node2][x],
        )
        cycle2.append(nearest_node2)
        visited.add(nearest_node2)

    return cycle1, cycle2


def greedy_algorithm_cycle(distance_matrix, n):

    # wybór pierwszego wierzchołka i najbliższego do niego
    start1 = random.randint(0, n - 1)
    cycle1 = [start1]
    visited = set(cycle1)

    distances_from_start1 = [distance_matrix[start1][i] for i in range(n)]
    closest_node1 = distances_from_start1.index(
        min(dist for i, dist in enumerate(distances_from_start1) if i not in visited)
    )
    cycle1.append(closest_node1)
    visited.add(closest_node1)

    # wybór drugiego wierzchołka i najbliższego do niego, zakładając, że są przynajmniej 4 wierzchołki
    start2 = distances_from_start1.index(
        max(dist for i, dist in enumerate(distances_from_start1) if i not in visited)
    )
    cycle2 = [start2]
    visited.add(start2)

    distances_from_start2 = [distance_matrix[start2][i] for i in range(n)]
    closest_node2 = distances_from_start2.index(
        min(dist for i, dist in enumerate(distances_from_start2) if i not in visited)
    )
    cycle2.append(closest_node2)
    visited.add(closest_node2)

    while len(cycle1) + len(cycle2) < n:
        best_new_node1 = None
        best_new_position1 = None
        best_increase1 = float("inf")

        best_new_node2 = None
        best_new_position2 = None
        best_increase2 = float("inf")

        # rozszerzanie cyklu 1
        for node in range(n):
            if node in visited:
                continue

            for i in range(len(cycle1)):
                increase = (
                    distance_matrix[cycle1[i]][node]
                    + distance_matrix[node][cycle1[(i + 1) % len(cycle1)]]
                    - distance_matrix[cycle1[i]][cycle1[(i + 1) % len(cycle1)]]
                )

                if increase < best_increase1:
                    best_increase1 = increase
                    best_new_node1 = node
                    best_new_position1 = i

        cycle1.insert(best_new_position1 + 1, best_new_node1)
        visited.add(best_new_node1)

        # rozszerzanie cyklu 2
        for node in range(n):
            if node in visited:
                continue

            for i in range(len(cycle2)):
                increase = (
                    distance_matrix[cycle2[i]][node]
                    + distance_matrix[node][cycle2[(i + 1) % len(cycle2)]]
                    - distance_matrix[cycle2[i]][cycle2[(i + 1) % len(cycle2)]]
                )

                if increase < best_increase2:
                    best_increase2 = increase
                    best_new_node2 = node
                    best_new_position2 = i

        cycle2.insert(best_new_position2 + 1, best_new_node2)
        visited.add(best_new_node2)

        # if len(cycle1) % 10 == 0:
        #     plot_solution(cycle1, cycle2, coordinates)

    return cycle1, cycle2


def cycle_length(cycle, distance_matrix):
    length = 0
    for i in range(len(cycle)):
        length += distance_matrix[cycle[i]][cycle[(i + 1) % len(cycle)]]
    return length


def plot_solution(cycle1, cycle2, coordinates):
    cycle1_coords = [coordinates[i] for i in cycle1]
    cycle2_coords = [coordinates[i] for i in cycle2]

    cycle1_coords.append(cycle1_coords[0])
    cycle2_coords.append(cycle2_coords[0])

    plt.figure(figsize=(10, 10))

    cycle1_x, cycle1_y = zip(*cycle1_coords)
    plt.plot(cycle1_x, cycle1_y, marker="o", label="Cykl 1", color="b")

    cycle2_x, cycle2_y = zip(*cycle2_coords)
    plt.plot(cycle2_x, cycle2_y, marker="o", label="Cykl 2", color="r")

    # Rysowanie wszystkich wierzchołków
    all_x, all_y = zip(*coordinates)
    plt.scatter(all_x, all_y, color="black", zorder=5)

    # Dodanie etykiety wierzchołków
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i), fontsize=8, ha="right")

    plt.title("Wizualizacja cykli")
    plt.legend()
    plt.show()


file_path = "kroa200.tsp"
distance_matrix, coordinates, n = read_instance(file_path)

cycle1, cycle2 = greedy_algorithm_cycle(distance_matrix, n)

length1 = cycle_length(cycle1, distance_matrix)
length2 = cycle_length(cycle2, distance_matrix)
print(f"Długość cyklu 1: {length1}")
print(f"Długość cyklu 2: {length2}")
print(f"Łączna długość: {length1 + length2}")

plot_solution(cycle1, cycle2, coordinates)
