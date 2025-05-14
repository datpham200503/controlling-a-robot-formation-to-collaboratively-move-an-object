#include <cstdint>
#include <limits>
#include <queue>
#include <vector>
#include <cmath>

constexpr double INF = std::numeric_limits<double>::infinity();

namespace graph {
namespace bidirectional_dijkstra {

void addEdge(std::vector<std::vector<std::pair<uint64_t, double>>> *adj1,
             std::vector<std::vector<std::pair<uint64_t, double>>> *adj2,
             uint64_t u, uint64_t v, double w, uint64_t edge_idx) {
    (*adj1)[u].push_back(std::make_pair(v, w));
    (*adj2)[v].push_back(std::make_pair(u, w));
}

double Shortest_Path_Distance(
    const std::vector<uint64_t> &workset_,
    const std::vector<std::vector<double>> &distance_,
    uint64_t *best_vertex) {
    double distance = INF;
    *best_vertex = UINT64_MAX;
    for (uint64_t i : workset_) {
        if (distance_[0][i] + distance_[1][i] < distance) {
            distance = distance_[0][i] + distance_[1][i];
            *best_vertex = i;
        }
    }
    return distance;
}

void ReconstructPath(
    const std::vector<std::vector<uint64_t>> &parent,
    const std::vector<std::vector<uint64_t>> &edge_idx,
    uint64_t s, uint64_t t, uint64_t best_vertex,
    uint64_t *path, uint64_t *path_edges, uint64_t *path_len) {
    std::vector<uint64_t> forward_path, forward_edges;
    std::vector<uint64_t> backward_path, backward_edges;

    // Reconstruct forward path from s to best_vertex
    uint64_t v = best_vertex;
    while (v != UINT64_MAX && v != s) {
        forward_path.push_back(v);
        forward_edges.push_back(edge_idx[0][v]);
        v = parent[0][v];
    }
    if (v == s) {
        forward_path.push_back(s);
    }

    // Reconstruct backward path from t to best_vertex
    v = best_vertex;
    while (v != UINT64_MAX && v != t) {
        backward_path.push_back(v);
        backward_edges.push_back(edge_idx[1][v]);
        v = parent[1][v];
    }
    if (v == t) {
        backward_path.push_back(t);
    }

    // Combine paths
    *path_len = 0;
    // Add forward path in reverse order (from s to best_vertex)
    for (int64_t i = forward_path.size() - 1; i >= 0; i--) {
        path[*path_len] = forward_path[i];
        if (i > 0) {
            path_edges[*path_len] = forward_edges[i - 1];
        }
        (*path_len)++;
    }
    // Add backward path (from best_vertex to t, excluding best_vertex)
    for (size_t i = 1; i < backward_path.size(); i++) {
        path[*path_len] = backward_path[i];
        path_edges[*path_len - 1] = backward_edges[i - 1];
        (*path_len)++;
    }
}

double Bidijkstra(std::vector<std::vector<std::pair<uint64_t, double>>> *adj1,
                  std::vector<std::vector<std::pair<uint64_t, double>>> *adj2,
                  uint64_t s, uint64_t t,
                  uint64_t *path, uint64_t *path_edges, uint64_t *path_len) {
    uint64_t n = adj1->size();
    std::vector<std::vector<double>> dist(2, std::vector<double>(n, INF));
    std::vector<std::vector<uint64_t>> parent(2, std::vector<uint64_t>(n, UINT64_MAX));
    std::vector<std::vector<uint64_t>> edge_idx(2, std::vector<uint64_t>(n, UINT64_MAX));
    std::vector<std::priority_queue<std::pair<double, uint64_t>,
                                   std::vector<std::pair<double, uint64_t>>,
                                   std::greater<std::pair<double, uint64_t>>>> pq(2);
    std::vector<uint64_t> workset;
    std::vector<bool> visited(n, false);

    pq[0].push(std::make_pair(0.0, s));
    dist[0][s] = 0.0;
    pq[1].push(std::make_pair(0.0, t));
    dist[1][t] = 0.0;

    uint64_t best_vertex = UINT64_MAX;

    while (true) {
        if (pq[0].empty()) break;
        uint64_t currentNode = pq[0].top().second;
        double currentDist = pq[0].top().first;
        pq[0].pop();

        for (size_t i = 0; i < (*adj1)[currentNode].size(); i++) {
            auto edge = (*adj1)[currentNode][i];
            uint64_t nextNode = edge.first;
            double weight = edge.second;
            if (currentDist + weight < dist[0][nextNode]) {
                dist[0][nextNode] = currentDist + weight;
                parent[0][nextNode] = currentNode;
                edge_idx[0][nextNode] = i;
                pq[0].push(std::make_pair(dist[0][nextNode], nextNode));
            }
        }
        workset.push_back(currentNode);
        if (visited[currentNode]) {
            double dist = Shortest_Path_Distance(workset, dist, &best_vertex);
            if (dist < INF) {
                ReconstructPath(parent, edge_idx, s, t, best_vertex, path, path_edges, path_len);
                return dist;
            }
        }
        visited[currentNode] = true;

        if (pq[1].empty()) break;
        currentNode = pq[1].top().second;
        currentDist = pq[1].top().first;
        pq[1].pop();

        for (size_t i = 0; i < (*adj2)[currentNode].size(); i++) {
            auto edge = (*adj2)[currentNode][i];
            uint64_t nextNode = edge.first;
            double weight = edge.second;
            if (currentDist + weight < dist[1][nextNode]) {
                dist[1][nextNode] = currentDist + weight;
                parent[1][nextNode] = currentNode;
                edge_idx[1][nextNode] = i;
                pq[1].push(std::make_pair(dist[1][nextNode], nextNode));
            }
        }
        workset.push_back(currentNode);
        if (visited[currentNode]) {
            double dist = Shortest_Path_Distance(workset, dist, &best_vertex);
            if (dist < INF) {
                ReconstructPath(parent, edge_idx, s, t, best_vertex, path, path_edges, path_len);
                return dist;
            }
        }
        visited[currentNode] = true;
    }
    *path_len = 0;
    return -1.0;
}

} // namespace bidirectional_dijkstra
} // namespace graph

extern "C" {
double bidirectional_dijkstra(uint64_t n, uint64_t m, uint64_t* edges, double* weights, uint64_t s, uint64_t t,
                             uint64_t* path, uint64_t* path_edges, uint64_t* path_len) {
    std::vector<std::vector<std::pair<uint64_t, double>>> adj1(n);
    std::vector<std::vector<std::pair<uint64_t, double>>> adj2(n);

    for (uint64_t i = 0; i < m; ++i) {
        uint64_t u = edges[2 * i];
        uint64_t v = edges[2 * i + 1];
        double w = weights[i];
        graph::bidirectional_dijkstra::addEdge(&adj1, &adj2, u, v, w, i);
    }

    return graph::bidirectional_dijkstra::Bidijkstra(&adj1, &adj2, s, t, path, path_edges, path_len);
}
}