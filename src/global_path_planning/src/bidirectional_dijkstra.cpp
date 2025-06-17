#include <cstdint>
#include <limits>
#include <queue>
#include <vector>
#include <map>

constexpr double INF = std::numeric_limits<double>::infinity();

struct Edge {
    uint64_t u, v; 
    double w;      
    uint64_t polytope_idx; 
};

struct Polytope {
    uint64_t id; 
};


struct PathResult {
    std::vector<uint64_t> T; 
    std::vector<Polytope> P; 
};


bool is_connected(const std::vector<std::vector<std::pair<uint64_t, double>>>& adj,
                  uint64_t z_s, uint64_t z_g, uint64_t n) {
    std::vector<bool> visited(n, false);
    std::vector<uint64_t> stack = {z_s};
    visited[z_s] = true;

    while (!stack.empty()) {
        uint64_t u = stack.back();
        stack.pop_back();
        if (u == z_g) return true;
        for (const auto& edge : adj[u]) {
            uint64_t v = edge.first;
            if (!visited[v]) {
                visited[v] = true;
                stack.push_back(v);
            }
        }
    }
    return false;
}


PathResult shortestPath(uint64_t n, const std::vector<Edge>& edges,
                        const std::vector<Polytope>& polytopes,
                        uint64_t z_s, uint64_t z_g) {

    std::vector<std::vector<std::pair<uint64_t, double>>> adj1(n);
    std::vector<std::vector<std::pair<uint64_t, double>>> adj2(n);
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> edge_map;


    for (uint64_t i = 0; i < edges.size(); ++i) {
        const Edge& e = edges[i];
        adj1[e.u].push_back({e.v, e.w});
        adj2[e.v].push_back({e.u, e.w});
        edge_map[{e.u, e.v}] = i;
    }


    if (!is_connected(adj1, z_s, z_g, n)) {
        return {{}, {}};
    }

    // Khởi tạo Dijkstra
    std::vector<std::vector<double>> dist(2, std::vector<double>(n, INF));
    std::vector<std::vector<uint64_t>> parent(2, std::vector<uint64_t>(n, UINT64_MAX));
    std::vector<std::priority_queue<std::pair<double, uint64_t>,
                                   std::vector<std::pair<double, uint64_t>>,
                                   std::greater<std::pair<double, uint64_t>>>> pq(2);
    std::vector<uint64_t> workset;
    std::vector<bool> visited(n, false);

    pq[0].push({0.0, z_s});
    dist[0][z_s] = 0.0;
    pq[1].push({0.0, z_g});
    dist[1][z_g] = 0.0;

    uint64_t best_vertex = UINT64_MAX;
    double shortest_dist = INF;

    // Bidirectional Dijkstra
    while (!pq[0].empty() || !pq[1].empty()) {
        if (!pq[0].empty()) {
            uint64_t u = pq[0].top().second;
            double d = pq[0].top().first;
            pq[0].pop();
            for (const auto& edge : adj1[u]) {
                uint64_t v = edge.first;
                double w = edge.second;
                if (d + w < dist[0][v]) {
                    dist[0][v] = d + w;
                    parent[0][v] = u;
                    pq[0].push({dist[0][v], v});
                }
            }
            workset.push_back(u);
            if (visited[u]) {
                for (uint64_t v : workset) {
                    if (dist[0][v] + dist[1][v] < shortest_dist) {
                        shortest_dist = dist[0][v] + dist[1][v];
                        best_vertex = v;
                    }
                }
                if (shortest_dist < INF) break;
            }
            visited[u] = true;
        }

        if (!pq[1].empty()) {
            uint64_t u = pq[1].top().second;
            double d = pq[1].top().first;
            pq[1].pop();
            for (const auto& edge : adj2[u]) {
                uint64_t v = edge.first;
                double w = edge.second;
                if (d + w < dist[1][v]) {
                    dist[1][v] = d + w;
                    parent[1][v] = u;
                    pq[1].push({dist[1][v], v});
                }
            }
            workset.push_back(u);
            if (visited[u]) {
                for (uint64_t v : workset) {
                    if (dist[0][v] + dist[1][v] < shortest_dist) {
                        shortest_dist = dist[0][v] + dist[1][v];
                        best_vertex = v;
                    }
                }
                if (shortest_dist < INF) break;
            }
            visited[u] = true;
        }
    }

    if (best_vertex == UINT64_MAX) {
        return {{}, {}};
    }

    // Tái tạo đường đi
    std::vector<uint64_t> forward_path, backward_path;
    uint64_t v = best_vertex;
    while (v != UINT64_MAX && v != z_s) {
        forward_path.push_back(v);
        v = parent[0][v];
    }
    if (v == z_s) forward_path.push_back(z_s);

    v = best_vertex;
    while (v != UINT64_MAX && v != z_g) {
        backward_path.push_back(v);
        v = parent[1][v];
    }
    if (v == z_g) backward_path.push_back(z_g);

    std::vector<uint64_t> T;
    for (int64_t i = forward_path.size() - 1; i >= 0; i--) {
        T.push_back(forward_path[i]);
    }
    for (size_t i = 1; i < backward_path.size(); i++) {
        T.push_back(backward_path[i]);
    }


    std::vector<Polytope> P;
    for (size_t i = 1; i < T.size(); i++) {
        auto it = edge_map.find({T[i-1], T[i]});
        if (it != edge_map.end()) {
            uint64_t edge_idx = it->second;
            uint64_t poly_idx = edges[edge_idx].polytope_idx;
            if (poly_idx < polytopes.size()) {
                P.push_back(polytopes[poly_idx]);
            }
        }
    }

    return {T, P};
}


extern "C" {

    void shortest_path(uint64_t n, // Số đỉnh
                       uint64_t m, // Số cạnh
                       uint64_t* edge_data, // Mảng [u1, v1, u2, v2, ...]
                       double* weights, // Mảng trọng số
                       uint64_t* polytope_ids, // Mảng chỉ số đa diện
                       uint64_t z_s, // Đỉnh bắt đầu
                       uint64_t z_g, // Đỉnh kết thúc
                       uint64_t* T_out, // Mảng đầu ra cho T
                       uint64_t* P_out, // Mảng đầu ra cho P
                       uint64_t* T_len, // Độ dài T
                       uint64_t* P_len) { // Độ dài P
        // Tạo vector cạnh
        std::vector<Edge> edges(m);
        for (uint64_t i = 0; i < m; ++i) {
            edges[i] = {edge_data[2*i], edge_data[2*i+1], weights[i], polytope_ids[i]};
        }

        // Tạo vector đa diện
        std::vector<Polytope> polytopes(m);
        for (uint64_t i = 0; i < m; ++i) {
            polytopes[i] = {polytope_ids[i]};
        }

        PathResult result = shortestPath(n, edges, polytopes, z_s, z_g);

        *T_len = result.T.size();
        *P_len = result.P.size();
        for (uint64_t i = 0; i < *T_len; ++i) {
            T_out[i] = result.T[i];
        }
        for (uint64_t i = 0; i < *P_len; ++i) {
            P_out[i] = result.P[i].id;
        }
    }
}