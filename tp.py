import pickle
from graph import Graph
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set
import heapq
from collections import deque
from queue import Queue, deque
import time

# PROBLEMA: TARDA MUCHO EN CORRER, NO SE SI ES PORQUE ESTA MAL IMPLEMENTADO O PORQUE EL GRAFO ES MUY GRANDE
# ALSO: imprime tipo en loop el coso de loading q antes no hacia ups creo que es x culpa de mi func de dfs - ES HASTA EL 36 Y DESP VA AL 56 DIRECTO WHAT. el loop vuelve en el 68 porciento
# ir comentando main para probar cada punto, ver que pasa y lo q esta abajo entre 3 comillas me lo pusieron los profes no editar

"""
Lista de productos

Ejemplo de producto:
{'id': 2,
 'title': 'Candlemas: Feast of Flames',
 'group': 'Book',
 'categories': ['Books[283155]->Subjects[1000]->Religion & Spirituality[22]->Earth-Based Religions[12472]->Wicca[12484]',
  'Books[283155]->Subjects[1000]->Religion & Spirituality[22]->Earth-Based Religions[12472]->Witchcraft[12486]'],
 'reviewers': [('A11NCO6YTE4BTJ', 5),
  ('A9CQ3PLRNIR83', 4),
  ('A13SG9ACZ9O5IM', 5),
  ('A1BDAI6VEYMAZA', 5),
  ('A2P6KAWXJ16234', 4),
  ('AMACWC3M7PQFR', 4),
  ('A3GO7UV9XX14D8', 4),
  ('A1GIL64QK68WKL', 5),
  ('AEOBOF2ONQJWV', 5),
  ('A3IGHTES8ME05L', 5),
  ('A1CP26N8RHYVVO', 1),
  ('ANEIANH0WAT9D', 5)]}
"""

def create_graph():
    #me la dieron 
    with open('products.pickle', 'rb') as file:
        products = pickle.load(file) 
    
    graph = Graph()

    print("Loading")
    for p in tqdm(products):
        graph.add_vertex(str(p["id"]), data={'title': p['title'],
                                            'group': p['group'],
                                            'categories': p['categories']})
        for reviewer, score in p['reviewers']:
            if not graph.vertex_exists(reviewer):
                graph.add_vertex(reviewer)
            graph.add_edge(reviewer, str(p["id"]), score)
            graph.add_edge(str(p["id"]), reviewer, score)
    return graph

# *******************    ALGORITMOS Y FUNCIONES AUXILIARES  ******************* #

def dijkstra(graph, start_vertex):
        distances = {vertex: float('infinity') for vertex in graph.get_vertexes()}
        distances[start_vertex] = 0
        priority_queue = [(0, start_vertex)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            for neighbor in graph.get_neighbors(current_vertex):
                weight = graph.get_edge_data(current_vertex, neighbor)
                distance = distances[current_vertex] + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances

def dfs(graph, vertex, work):
    visited = set()
    stack = deque()
    stack.append(vertex)
    visited.add(vertex)
    while len(stack) != 0:
        current = stack.pop()
        for x in graph.get_neighbors(current):
            if x not in visited:
                work(x)
                stack.append(x)
                visited.add(x)


def bfs(graph, vertex):
    queue = []
    queue.append(vertex)
    visited_nodes = set()
    visited_nodes.add(vertex)
    all_distances = {vertex: 0}

    while len(queue) != 0:
        current_vertex = queue.pop(0)
        distance = all_distances[current_vertex] + 1

        for neighbor in graph.get_neighbors_component(current_vertex):
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                all_distances[neighbor] = distance
                queue.append(neighbor)

    return all_distances


def add_vertex_to_set(set, vertex):
    set.add(vertex)


def add_vertex_to_list(list, vertex):
    list.append(vertex)


def check_if_vertex_in_connected_component(connected_components, vertex):
    for element in connected_components:
        if vertex in element:
            return True
    return False


def get_connected_components(graph):
    vertexes = graph.get_vertexes()
    connected_components = []
    for vertex in tqdm(vertexes):
        if not (check_if_vertex_in_connected_component(connected_components, vertex)):
            connected_component = set()
            dfs(graph, vertex, lambda x: add_vertex_to_set(connected_component, x))
            connected_components.append(connected_component)
    return connected_components

def dfs_shortest_path(graph, start_vertex, target_vertex, visited=None, current_distance=0, min_distance=float('inf')):
    if visited is None:
        visited = set()

    visited.add(start_vertex)

    if start_vertex == target_vertex:
        return min(current_distance, min_distance)

    for neighbor in graph.get_neighbors(start_vertex):
        if neighbor not in visited:
            min_distance = dfs_shortest_path(graph, neighbor, target_vertex, visited.copy(), current_distance + 1, min_distance)

    return min_distance

def min_separation(graph, vertex1, vertex2):
    separation = dfs_shortest_path(graph, vertex1, vertex2)

    if separation == float('inf'):
        return -1  # Indicating no connection found between books.
    else:
        return separation

def minimize_inverse_rating(graph, start_vertex, end_vertex):
    distances = {vertex: float('infinity') for vertex in graph._graph}
    distances[start_vertex] = 0
    previous_vertices = {vertex: None for vertex in graph._graph}

    priority_queue = [(0, start_vertex)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor in graph.get_neighbors(current_vertex):
            rating = 1 / graph.get_edge_data(current_vertex, neighbor)
            distance = distances[current_vertex] + rating

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor))

    path = []
    current_vertex = end_vertex
    while previous_vertices[current_vertex] is not None:
        path.insert(0, current_vertex)
        current_vertex = previous_vertices[current_vertex]

    path.insert(0, start_vertex)
    return path

def degree_of_separation( graph, vertex1, vertex2):
    return dfs_shortest_path(graph, vertex1, vertex2)


# *******************    EJERCICIOS & MAIN  ******************* #
def punto1(graph):
    '''
        utilizo el algoritmo de bfs para encontrar  los distintos componentes conexos y luego de ordenarlos,
        me quedo con el segundo mas grande y el mas chico.
    '''
    connected_components = get_connected_components(graph)
    is_connected = len(connected_components) == 1

    print("El grafo es conexo:", is_connected)
    print("La cantidad de componentes conexos es ", len(connected_components))

    sorted_components = sorted(connected_components, key=len, reverse=True)
    largest_component = sorted_components[0]
    smallest_component = sorted_components[-1]
    print("El largo del componente conexo más grande es: ", len(largest_component))
    print("El largo del componente conexo más chico es: ", len(smallest_component))
    print("Vértices en la componente conexa más chico: ", smallest_component)

def punto2(graph):
    book1, book2 = "2", "3"
    degree = degree_of_separation(graph, book1, book2)
    print("El grado de separación entre los libros ", book1, " y ", book2, " es: ", degree)

def punto3(graph):
    target_book_vertex = "OpenSocietyAndItsEnemiesVertex"
    current_book_vertex = "YourCurrentBookVertex"
    path = minimize_inverse_rating(graph, current_book_vertex, target_book_vertex)

    # Imprime el camino sugerido
    print(f"Camino sugerido para leer '{graph.get_vertex_data(target_book_vertex)['title']}':")
    for vertex in path:
        title = graph.get_vertex_data(vertex)['title']
        print(f" - {title}")

    return path   

def punto4(graph):
    connected_components = get_connected_components(graph)
    is_connected = len(connected_components) == 1

    sorted_components = sorted(connected_components, key=len, reverse=True)
    largest_component = sorted_components[0]

    diameter = 0
    start = time.time()
    for vertex in tqdm(largest_component):
        for neighbor in largest_component:
            if vertex != neighbor:
                diameter = max(diameter, degree_of_separation(graph, vertex, neighbor))
    end = time.time()
    print("El diámetro del grafo es: ", diameter)
    print("El tiempo de ejecución fue de: ", end - start)

def main(): 

    graph = create_graph()
    related_components = get_connected_components(graph)
    sorted_components = sorted(related_components, key=len, reverse=True)

    print("============================================================")
    print(" ** Punto 1 ** ")
    punto1(graph)
    print(" ** Punto 2 ** ")
    punto2(graph) 
    print(" ** Punto 3 ** ")
    punto3(graph)
    print(" ** Punto 4 ** ")
    punto4(graph)


if __name__ == "__main__":
    main()
