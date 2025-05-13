import matplotlib.pyplot as plt
import networkx as nx

# Class definition for node to be added to graph
class AdjacentNode:
    def __init__(self, vertex):
        self.vertex = vertex
        self.next = None

# BidirectionalSearch implementation
class BidirectionalSearch:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [None] * self.vertices

        self.src_queue = []
        self.dest_queue = []

        self.src_visited = [False] * self.vertices
        self.dest_visited = [False] * self.vertices

        self.src_parent = [None] * self.vertices
        self.dest_parent = [None] * self.vertices

        # Dùng NetworkX để vẽ đồ thị
        self.nx_graph = nx.Graph()

    def add_edge(self, src, dest):
        node = AdjacentNode(dest)
        node.next = self.graph[src]
        self.graph[src] = node

        node = AdjacentNode(src)
        node.next = self.graph[dest]
        self.graph[dest] = node

        self.nx_graph.add_edge(src, dest)

    def bfs(self, direction='forward'):
        if direction == 'forward':
            current = self.src_queue.pop(0)
            connected_node = self.graph[current]
            while connected_node:
                vertex = connected_node.vertex
                if not self.src_visited[vertex]:
                    self.src_queue.append(vertex)
                    self.src_visited[vertex] = True
                    self.src_parent[vertex] = current
                connected_node = connected_node.next
        else:
            current = self.dest_queue.pop(0)
            connected_node = self.graph[current]
            while connected_node:
                vertex = connected_node.vertex
                if not self.dest_visited[vertex]:
                    self.dest_queue.append(vertex)
                    self.dest_visited[vertex] = True
                    self.dest_parent[vertex] = current
                connected_node = connected_node.next

    def is_intersecting(self):
        for i in range(self.vertices):
            if self.src_visited[i] and self.dest_visited[i]:
                return i
        return -1

    def build_path(self, intersecting_node, src, dest):
        path = [intersecting_node]
        i = intersecting_node
        while i != src:
            path.append(self.src_parent[i])
            i = self.src_parent[i]
        path = path[::-1]
        i = intersecting_node
        while i != dest:
            i = self.dest_parent[i]
            path.append(i)
        return path

    def visualize_graph(self, path=None):
        pos = nx.spring_layout(self.nx_graph, seed=42)
        plt.figure(figsize=(10, 6))
        nx.draw(self.nx_graph, pos, with_labels=True, node_color='lightblue', node_size=700)

        if path:
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.nx_graph, pos, edgelist=edges, edge_color='red', width=3)
            nx.draw_networkx_nodes(self.nx_graph, pos, nodelist=path, node_color='red')

        plt.title("Bidirectional Search Path Visualization")
        plt.show()

    def bidirectional_search(self, src, dest, show=True):
        self.src_queue.append(src)
        self.src_visited[src] = True
        self.src_parent[src] = -1

        self.dest_queue.append(dest)
        self.dest_visited[dest] = True
        self.dest_parent[dest] = -1

        while self.src_queue and self.dest_queue:
            self.bfs('forward')
            self.bfs('backward')

            intersecting_node = self.is_intersecting()
            if intersecting_node != -1:
                path = self.build_path(intersecting_node, src, dest)
                if show:
                    print(f"Path exists between {src} and {dest}")
                    print(f"Intersection at : {intersecting_node}")
                    print("***** Path *****")
                    print(" -> ".join(map(str, path)))
                    self.visualize_graph(path)
                else:
                    print(f"Path exists between {src} and {dest}")
                    print(f"Intersection at : {intersecting_node}")
                    print("***** Path *****")
                    print(" -> ".join(map(str, path)))
                return path

        if show:
            print(f"No path exists between {src} and {dest}")
            self.visualize_graph()
        return -1


# ----------- Demo chạy -----------
if __name__ == '__main__':
    n = 15
    src = 0
    dest = 11

    graph = BidirectionalSearch(n)
    graph.add_edge(0, 4)
    graph.add_edge(1, 4)
    graph.add_edge(2, 5)
    graph.add_edge(3, 5)
    graph.add_edge(4, 6)
    graph.add_edge(5, 6)
    graph.add_edge(6, 7)
    graph.add_edge(7, 8)
    graph.add_edge(8, 9)
    graph.add_edge(8, 10)
    graph.add_edge(9, 11)
    graph.add_edge(9, 12)
    graph.add_edge(10, 13)
    graph.add_edge(10, 14)
    graph.add_edge(12, 6)

    graph.bidirectional_search(src, dest, show=True)
