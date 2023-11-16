from typing import Dict, List, Tuple, Optional
from graph import Graph
import heapq


class GraphUtils:
    @staticmethod
    def dijkstra(graph: Graph, start_vertex: str) -> Dict[str, Tuple[Optional[str], int]]:
        """
        Dijkstra's algorithm for finding shortest paths from a start vertex to all other vertices in a graph.
        :param graph: The graph
        :param start_vertex: The starting vertex
        :return: Dictionary containing the shortest path and distance to each vertex from the start vertex
        """
        distances = {vertex: float('infinity') for vertex in graph._graph}
        distances[start_vertex] = 0
        previous_vertices = {vertex: None for vertex in graph._graph}

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
                    previous_vertices[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

        shortest_paths = {vertex: (previous, distance) for vertex, (previous, distance) in
                          zip(distances.keys(), zip(previous_vertices.values(), distances.values()))}

        return shortest_paths

    @staticmethod
    def dfs(graph: Graph, start_vertex: str) -> List[str]:
        """
        Depth-first search on a graph.
        :param graph: The graph
        :param start_vertex: The starting vertex
        :return: List of vertices visited in DFS order
        """
        visited = set()
        dfs_order = []

        def dfs_recursive(vertex):
            visited.add(vertex)
            dfs_order.append(vertex)
            for neighbor in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs_recursive(neighbor)

        dfs_recursive(start_vertex)
        return dfs_order
