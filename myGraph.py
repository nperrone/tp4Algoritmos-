from graph import Graph
from graph_utils import GraphUtils



def punto1(graph: Graph):
    def is_connected(graph: Graph) -> bool:
        # Implementación para verificar si el grafo es conexo
        pass

    def connected_components(graph: Graph) -> Tuple[int, int, List[str], int, List[str]]:
        # Implementación para obtener componentes conexas y sus propiedades
        pass

    def print_component_properties(component_id: int, component: List[str]):
        # Función para imprimir propiedades de una componente conexa
        pass

    connected = is_connected(graph)
    num_components, max_size, max_component, min_size, min_component = connected_components(graph)

    print(f"El grafo es conexo: {connected}")
    print(f"Cantidad de componentes conexas: {num_components}")
    print(f"Tamaño de la componente más grande: {max_size}")
    print(f"Contenido y cantidad de vértices de la componente más grande:")
    print_component_properties(1, max_component)
    print(f"Tamaño de la componente más pequeña: {min_size}")
    print(f"Contenido y cantidad de vértices de la componente más pequeña:")
    print_component_properties(2, min_component)


# Ahora puedes utilizar estas funciones en tu código principal

# Ejemplo de cómo usar las funciones
if __name__ == "__main__":
    graph_utils = GraphUtils()

    # Puedes llamar a las funciones necesarias aquí para realizar las operaciones requeridas
    # graph_data = read_data()  # Implementa esta función si es necesario

    # Luego, construyes el grafo y realizas el análisis
    # grafo = Graph()
    # Luego de cargar los datos en el grafo, puedes llamar a la función punto1
    # punto1(grafo)
