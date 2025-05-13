import sys
import math
import heapq
import matplotlib.pyplot as plt

class AStar:
    def __init__(self, n, adj, cost, x, y):
        self.n = n
        self.adj = adj
        self.cost = cost
        self.inf = n*10**6
        self.f = [[self.inf]*n, [self.inf]*n]       # forward & backward; f(v) = g(v) + h(v); these are new distances, with potential h (h(v) is computed on the fly); Dijkstra UCS now works with them (this is self.distance in Dijkstra); self.f can be a map (dictionary)
        self.g = [[self.inf]*n, [self.inf]*n]       # forward & backward;  this is the known part of the distance; h is the heuristic part; this is the true distance from starting vertex to v; self.g can be a map (dictionary)
        self.closedF = set()                        # forward closed set (processed nodes)
        self.closedR = set()                        # backward closed set (processed nodes)
        self.valid = [[True] * n, [True] * n]       # is vertex (name) valid or not - it's valid while name (vertex) is in open set (in heap)
        self.parent = [[-1]*n, [-1]*n]
        # Coordinates of the nodes
        self.x = x
        self.y = y

    def clear(self):
        """ Reinitialize the data structures for the next query after the previous query. """
        self.f = [[self.inf]*n, [self.inf]*n]
        self.g = [[self.inf]*n, [self.inf]*n]
        self.closedF.clear()
        self.closedR.clear()
        self.valid = [[True] * n, [True] * n]
        self.parent = [[-1]*self.n, [-1]*self.n]

    def heur(self, s, t, v):
        """ Simple Euclidean distance heuristic """
        dist_s = math.sqrt((self.x[v] - self.x[s])**2 + (self.y[v] - self.y[s])**2)
        dist_t = math.sqrt((self.x[v] - self.x[t])**2 + (self.y[v] - self.y[t])**2)
        return dist_s, dist_t

    def visit(self, open, side, name, s, t):
        closed = self.closedF if side == 0 else self.closedR
        for i in range(len(self.adj[side][name])):
            neighbor = self.adj[side][name][i]
            if neighbor in closed:     # also ok: if (self.f[neighbor], neighbor) in closed:
                continue
            temp_g = self.g[side][name] + self.cost[side][name][i]
            if (self.f[side][neighbor], neighbor) not in open[side]:
                hf, hr = self.heur(s, t, neighbor)
                self.g[side][neighbor] = temp_g
                self.parent[side][neighbor] = name
                self.f[side][neighbor] = temp_g + (hf if side == 0 else hr)     # h depends on the side!
                heapq.heappush(open[side], (self.f[side][neighbor], neighbor))
                continue
            if self.g[side][neighbor] > temp_g:
                hf, hr = self.heur(s, t, neighbor)
                self.g[side][neighbor] = temp_g
                self.f[side][neighbor] = temp_g + (hf if side == 0 else hr)     # h depends on the side!
                heapq.heappush(open[side], (self.f[side][neighbor], neighbor))

    def reconstruct_path(self, u, s, t):
        path_forward = []
        while u != -1:
            path_forward.append(u)
            u = self.parent[0][u]
        path_forward.reverse()

        u = self.parent[1][path_forward[-1]]  # start from meeting point (avoid duplication)
        path_backward = []
        while u != -1:
            path_backward.append(u)
            u = self.parent[1][u]

        full_path = path_forward + path_backward
        return [x + 1 for x in full_path]

    def query(self, s, t):
        """ Returns the distance from s to t in the graph (-1 if there's no path). """
        self.clear()
        open = [[], []]  # Priority queues for forward and reverse searches

        # Initialize forward search
        hf, hr = self.heur(s, t, s)
        self.g[0][s] = 0
        self.f[0][s] = self.g[0][s] + hf
        heapq.heappush(open[0], (self.f[0][s], s))

        # Initialize reverse search
        hf, hr = self.heur(s, t, t)
        self.g[1][t] = 0
        self.f[1][t] = self.g[1][t] + hr
        heapq.heappush(open[1], (self.f[1][t], t))

        while open[0] or open[1]:
            best = None
            name = None

            # Forward search step
            while open[0]:
                best = heapq.heappop(open[0])
                name = best[1]
                if self.valid[0][name]:
                    self.valid[0][name] = False
                    break
            self.visit(open, 0, name, s, t)  # forward

            if name in self.closedR:
                break
            self.closedF.add(name)

            # Reverse search step
            best = None
            name = None
            while open[1]:
                best = heapq.heappop(open[1])
                name = best[1]
                if self.valid[1][name]:
                    self.valid[1][name] = False
                    break
            self.visit(open, 1, name, s, t)  # reverse

            if name in self.closedF:
                break
            self.closedR.add(name)

        distance = self.inf
        self.closedF = self.closedF | self.closedR  # Merge closed sets

        for u in self.closedF:
            if (self.g[0][u] + self.g[1][u] < distance):
                distance = self.g[0][u] + self.g[1][u]
                meet = u

        if distance < self.inf:
            path = self.reconstruct_path(meet, s, t)
            self.plot_graph(path)  # Call the function to plot the graph
            print("Path:", " → ".join(map(str, path)))
            return distance

        return -1


    def plot_graph(self, path=None):
        """ Plot the graph with nodes, edges, and the shortest path (if provided). """
        plt.figure(figsize=(8, 8))
        
        # Plot nodes
        plt.scatter(self.x, self.y, color='blue', zorder=5)
        for i in range(self.n):
            plt.text(self.x[i], self.y[i], str(i + 1), fontsize=12, ha='center', va='center', color='black', zorder=10)

        # Plot edges
        for i in range(self.n):
            for j in self.adj[0][i]:
                plt.plot([self.x[i], self.x[j]], [self.y[i], self.y[j]], color='grey', alpha=0.5, zorder=2)

        # Highlight the path
        if path:
            for i in range(len(path) - 1):
                u = path[i] - 1
                v = path[i + 1] - 1
                plt.plot([self.x[u], self.x[v]], [self.y[u], self.y[v]], color='red', linewidth=2, zorder=3)

        plt.title('Graph Visualization with Shortest Path')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    from io import StringIO
    input_data = """\
    6 8
    0 0
    1 0
    2 0
    2 1
    2 2
    0 2
    1 6 100
    2 3 1
    3 4 1
    4 5 1
    0 1 2
    1 4 1
    2 1 1
    3 6 1
    1
    1 6
    """
    input = StringIO(input_data).read

    data = list(map(int, input().split()))
    n, m = data[0:2]    # number of nodes, number of edges; nodes are numbered from 1 to n
    data = data[2:]
    x = [0] * n
    y = [0] * n
    adj = [[[] for _ in range(n)], [[] for _ in range(n)]]    # holds adjacency lists for every vertex in the graph; contains both forward and reverse arrays
    cost = [[[] for _ in range(n)], [[] for _ in range(n)]]   # holds weights of the edges; contains both forward and reverse arrays
    for i in range(n):
        x[i] = data[i << 1]
        y[i] = data[(i << 1) + 1]
    data = data[2*n:]
    for e in range(m):
        u = data[3*e]
        v = data[3*e+1]
        c = data[3*e+2]
        adj[0][u-1].append(v-1)
        cost[0][u-1].append(c)
        adj[1][v-1].append(u-1)
        cost[1][v-1].append(c)
    astar = AStar(n, adj, cost, x, y)
    data = data[3*m:]
    q = data[0]
    data = data[1:]
    for i in range(q):
        s = data[i << 1]
        t = data[(i << 1) + 1]
        print(astar.query(s-1, t-1))
