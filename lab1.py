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


def find_first_cycles(distance_matrix, n):

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

    return cycle1, cycle2, visited


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

    cycle1, cycle2, visited = find_first_cycles(distance_matrix, n)

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

    return cycle1, cycle2


def heuristic_algorithm_regret(distance_matrix, n):

    cycle1, cycle2, visited = find_first_cycles(distance_matrix, n)

    while len(cycle1) + len(cycle2) < n:
        best_new_node1 = None
        best_new_position1 = None
        best_regret1 = -float("inf")

        best_new_node2 = None
        best_new_position2 = None
        best_regret2 = -float("inf")

        # znalezienie wierzchołka z największym żalem dla cyklu 1
        for node in range(n):
            if node in visited:
                continue

            insertion_costs = []
            for i in range(len(cycle1)):
                cost = (
                    distance_matrix[cycle1[i]][node]
                    + distance_matrix[node][cycle1[(i + 1) % len(cycle1)]]
                    - distance_matrix[cycle1[i]][cycle1[(i + 1) % len(cycle1)]]
                )
                insertion_costs.append((cost, i))

            if len(insertion_costs) > 1:
                insertion_costs.sort(key=lambda x: x[0])
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                if regret > best_regret1:
                    best_regret1 = regret
                    best_new_node1 = node
                    best_new_position1 = insertion_costs[0][1]

        cycle1.insert(best_new_position1 + 1, best_new_node1)
        visited.add(best_new_node1)

        # znalezienie wierzchołka dla cyklu 2
        for node in range(n):
            if node in visited:
                continue

            insertion_costs = []
            for i in range(len(cycle2)):
                cost = (
                    distance_matrix[cycle2[i]][node]
                    + distance_matrix[node][cycle2[(i + 1) % len(cycle2)]]
                    - distance_matrix[cycle2[i]][cycle2[(i + 1) % len(cycle2)]]
                )
                insertion_costs.append((cost, i))

            if len(insertion_costs) > 1:
                insertion_costs.sort(key=lambda x: x[0])
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                if regret > best_regret2:
                    best_regret2 = regret
                    best_new_node2 = node
                    best_new_position2 = insertion_costs[0][1]

        cycle2.insert(best_new_position2 + 1, best_new_node2)
        visited.add(best_new_node2)

    return cycle1, cycle2


def heuristic_algorithm_regret_cycles(distance_matrix, n):

    # znajduje dwa najlepsze wierzchołki do wstawienia w podany cykl, liczy żal dla wierzchołków
    def find_best_node(cycle, distance_matrix, visited):
        nodes_increases = []

        # dla każdego wierzchołka
        for node in range(n):
            if node in visited:
                continue

            best_position_for_node = None
            best_regret_for_node = -float("inf")
            insertion_cost = []

            # dla każdego miejsca w cyklu
            for i in range(len(cycle)):
                cost = (
                    distance_matrix[cycle[i]][node]
                    + distance_matrix[node][cycle[(i + 1) % len(cycle)]]
                    - distance_matrix[cycle[i]][cycle[(i + 1) % len(cycle)]]
                )
                insertion_cost.append((cost, i))

            if len(insertion_cost) > 1:
                insertion_cost.sort(key=lambda x: x[0])
                regret = insertion_cost[1][0] - insertion_cost[0][0]
                if regret > best_regret_for_node:
                    best_regret_for_node = regret
                    best_position_for_node = insertion_cost[0][1]

            nodes_increases.append((node, best_regret_for_node, best_position_for_node))

        nodes_increases.sort(key=lambda x: x[1], reverse=True)

        return nodes_increases[:2]  # zwracamy dwa najlepsze wierzchołki

    cycle1, cycle2, visited = find_first_cycles(distance_matrix, n)

    while len(cycle1) + len(cycle2) < n:

        # znajdowanie najlepszych wierzchołków do wstawienia dla obu cykli
        best_new_node1 = find_best_node(cycle1, distance_matrix, visited)
        best_new_node2 = find_best_node(cycle2, distance_matrix, visited)

        # jeżeli najlepsze wierzchołki dla obu cykli są takie same
        if best_new_node1[0][0] == best_new_node2[0][0]:

            # żal na poziomie cykli
            regret1 = best_new_node1[1][1] - best_new_node1[0][1]
            regret2 = best_new_node2[1][1] - best_new_node2[0][1]

            if regret1 > regret2:
                cycle1.insert(best_new_node1[0][2] + 1, best_new_node1[0][0])
                visited.add(best_new_node1[0][0])
                cycle2.insert(best_new_node2[1][2] + 1, best_new_node2[1][0])
                visited.add(best_new_node2[1][0])
            else:
                cycle1.insert(best_new_node1[1][2] + 1, best_new_node1[1][0])
                visited.add(best_new_node1[1][0])
                cycle2.insert(best_new_node2[0][2] + 1, best_new_node2[0][0])
                visited.add(best_new_node2[0][0])

        # najlepsze wierzchołki dla obu cykli są różne
        else:
            cycle1.insert(best_new_node1[0][2] + 1, best_new_node1[0][0])
            visited.add(best_new_node1[0][0])
            cycle2.insert(best_new_node2[0][2] + 1, best_new_node2[0][0])
            visited.add(best_new_node2[0][0])

    return cycle1, cycle2


def heuristic_algorithm_regret_weighted(
    distance_matrix, n, weight_regret=1, weight_greedy=0.1 # żal to różnica pomiędzy odległościami, a greedy to odległość do najbliższego wierzchołka
):                  # dlatego różnica w wagach (inna skala)

    cycle1, cycle2, visited = find_first_cycles(distance_matrix, n)

    while len(cycle1) + len(cycle2) < n:
        best_new_node1 = None
        best_new_position1 = None
        best_score1 = -float("inf")

        best_new_node2 = None
        best_new_position2 = None
        best_score2 = -float("inf")

        # znalezienie wierzchołka z największym żalem ważonym dla cyklu 1
        for node in range(n):
            if node in visited:
                continue

            insertion_costs = []
            for i in range(len(cycle1)):
                cost = (
                    distance_matrix[cycle1[i]][node]
                    + distance_matrix[node][cycle1[(i + 1) % len(cycle1)]]
                    - distance_matrix[cycle1[i]][cycle1[(i + 1) % len(cycle1)]]
                )
                insertion_costs.append((cost, i))

            if len(insertion_costs) > 1:
                insertion_costs.sort(key=lambda x: x[0])
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                greedy_value = insertion_costs[0][0]

                score = weight_regret * regret + (-1) * weight_greedy * greedy_value

                if score > best_score1:
                    best_score1 = score
                    best_new_node1 = node
                    best_new_position1 = insertion_costs[0][1]

        cycle1.insert(best_new_position1 + 1, best_new_node1)
        visited.add(best_new_node1)

        # znalezienie wierzchołka dla cyklu 2
        for node in range(n):
            if node in visited:
                continue

            insertion_costs = []
            for i in range(len(cycle2)):
                cost = (
                    distance_matrix[cycle2[i]][node]
                    + distance_matrix[node][cycle2[(i + 1) % len(cycle2)]]
                    - distance_matrix[cycle2[i]][cycle2[(i + 1) % len(cycle2)]]
                )
                insertion_costs.append((cost, i))

            if len(insertion_costs) > 1:
                insertion_costs.sort(key=lambda x: x[0])
                regret = insertion_costs[1][0] - insertion_costs[0][0]
                greedy_value = insertion_costs[0][0]

                score = weight_regret * regret + (-1) * weight_greedy * greedy_value

                if score > best_score2:
                    best_score2 = regret
                    best_new_node2 = node
                    best_new_position2 = insertion_costs[0][1]

        cycle2.insert(best_new_position2 + 1, best_new_node2)
        visited.add(best_new_node2)

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

def sim(distance_matrix, coordinates, n,method):
    cycle1 =[]
    cycle2 =[]
    print(m)
    if method == "heuristic_algorithm_regret_cycles":
       cycle1, cycle2 = heuristic_algorithm_regret_cycles(distance_matrix, n) 
    elif method == "heuristic_algorithm_regret_weighted":
        cycle1, cycle2 = heuristic_algorithm_regret_weighted(distance_matrix, n)
    elif method == "greedy_algorithm_cycle":
        cycle1, cycle2 = greedy_algorithm_cycle(distance_matrix, n)
    elif method == "greedy_algorithm_nearest_neighbour":
        cycle1, cycle2 = greedy_algorithm_nearest_neighbour(distance_matrix, n)
    elif method == "heuristic_algorithm_regret":
        cycle1, cycle2 = heuristic_algorithm_regret(distance_matrix, n)

        
    length1 = cycle_length(cycle1, distance_matrix)
    length2 = cycle_length(cycle2, distance_matrix)
    return cycle1, cycle2, (length1+length2)

methods = [
 #   "heuristic_algorithm_regret_cycles",
   "heuristic_algorithm_regret_weighted",
 #   "greedy_algorithm_cycle",
 #   "greedy_algorithm_nearest_neighbour",
    "heuristic_algorithm_regret"
]
file_paths = ["kroB200.tsp", "kroA200.tsp"]
NUM_ITTER =10
results = {}
NUM_ITTER =10
for file in file_paths:
    distance_matrix, coordinates, n = read_instance(file)
    results[file] = {}

    for m in methods:
        best_cycle1, best_cycle2, best_result = None, None, float("inf")
        result_list = []

        for i in range(NUM_ITTER):
            print(f"Running {m} on {file}, iteration {i}")
            cycle1, cycle2, result = sim(distance_matrix, coordinates, n, m)
            result_list.append(result)

            if result < best_result:
                best_result = result
                best_cycle1, best_cycle2 = cycle1, cycle2

        results[file][m] = {
            "best_cycle1": best_cycle1,
            "best_cycle2": best_cycle2,
            "min_result": min(result_list),
            "max_result": max(result_list),
            "sum_results": sum(result_list)
        }

instances = list(results.keys())
header = f"{'Metoda':<40}" + "".join([f"{inst:<30}" for inst in instances])
table_lines = [header, "-" * len(header)]

for method in methods:
    row = f"{method:<40}"  # Nazwa metody
    for instance in instances:
        if method in results[instance]:  
            data = results[instance][method]
            avg = round(data["sum_results"] / NUM_ITTER, 2) if data["sum_results"] is not None else "N/A"
            min_val = data["min_result"] if data["min_result"] is not None else "N/A"
            max_val = data["max_result"] if data["max_result"] is not None else "N/A"
            row += f"{avg} ({min_val} – {max_val})".ljust(30)
        else:
            row += "Brak danych".ljust(30)
    table_lines.append(row)


# Zapis do pliku
output_file = "wyniki_tabela.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(table_lines))

# Wyświetlenie tabeli w konsoli
print("\n".join(table_lines))
print(f"\nWyniki zapisane do pliku: {output_file}")
for file, methods_results in results.items():
    print(f"\n=== Wyniki dla pliku: {file} ===")
    distance_matrix, coordinates, n = read_instance(file)
    for method, data in methods_results.items():
        print(f"\nMetoda: {method}")
        print(f" - Najlepszy wynik: {data['min_result']}")
        print(f" - Najgorszy wynik: {data['max_result']}")
        print(f" - Średnia wyników: {data['sum_results']/NUM_ITTER}")
        plot_solution(data['best_cycle1'],data['best_cycle2'],coordinates)

