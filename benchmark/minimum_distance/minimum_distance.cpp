#include "solver.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

using NodeId = std::string;
using NodeIdx = std::uint32_t;
using Distance = std::int32_t;
using Visit = std::pair<Distance, NodeIdx>;

struct Edge {
    NodeIdx to;
    Distance dist;
};

struct Graph {
    std::uint32_t nodes = 0;
    std::vector<std::vector<Edge>> edges;
    // TODO: std::stringをたくさんコピーしたくない
    std::map<NodeId, NodeIdx> id2idx;
    std::vector<NodeId> idx2id;
    void addNode(NodeId n)
    {
        nodes++;
        idx2id.push_back(n);
        id2idx[n] = nodes;
        edges.resize(nodes);
    }
    void addEdge(NodeIdx from, NodeIdx to, Distance d) { edges[from].emplace_back(Edge{to, d}); }
    std::uint32_t size() const { return nodes; }
};

std::tuple<Distance, std::vector<NodeIdx>> dijkstra(const Graph& g, NodeIdx s, NodeIdx t)
{
    std::vector<Distance> distance(g.nodes, std::numeric_limits<Distance>::max());
    std::priority_queue<Visit, std::vector<Visit>, std::greater<Visit>> que;

    distance[s] = 0;
    que.push({0, s});
    while (!que.empty()) {
        const auto [d, from] = que.top();
        que.pop();
        if (from == t) break;
        if (distance[from] < d) continue;
        for (const auto& [to, l] : g.edges[from]) {
            const auto dist = d + l;
            if (dist < distance[to]) {
                distance[to] = dist;
                que.push({dist, to});
            }
        }
    }

    if (distance[t] == std::numeric_limits<Distance>::max()) return {distance[t], {}};

    std::vector<NodeIdx> route;
    route.push_back(t);
    while (true) {
        const auto from = route.back();
        if (from == s) break;
        for (const auto& [to, l] : g.edges[from]) {
            if (distance[from] == distance[to] + l) {
                route.push_back(to);
                break;
            }
        }
    }
    std::reverse(route.begin(), route.end());
    return {distance[t], route};
}

Graph loadSGBWords(const std::string& filepath, std::size_t graph_size = std::numeric_limits<std::size_t>::max())
{
    std::ifstream ifs(filepath);
    if (!ifs.good()) {
        std::cerr << "Cannot open sgb-words.txt" << std::endl;
        exit(1);
    }
    Graph g;
    std::string word;
    std::size_t size = 0;
    while (std::getline(ifs, word)) {
        if (word.size() != 5) continue;
        g.addNode(word);
        size++;
        if (size == graph_size) break;
    }
    ifs.close();

    // construct edge
    // TODO: 辺のつながり、距離をどう定義するか良い感じに調整する
    const auto connect = [](const char* a, const char* b) {
        auto match = 0;
        for (auto i = 0; i < 5; ++i) {
            if (a[i] == b[i]) match++;
        }
        if (match >= 3)
            return 5 - match;
        else
            return 0;
    };
    for (std::uint32_t x = 0; x < g.size(); ++x) {
        for (std::uint32_t y = x + 1; y < g.size(); ++y) {
            if (const auto dist = connect(g.idx2id[x].c_str(), g.idx2id[y].c_str()); dist != 0) {
                g.addEdge(x, y, dist);
                g.addEdge(y, x, dist);
            }
        }
    }
    return g;
}

solver::LPSolver minimumDistance(const Graph& g, NodeIdx s, NodeIdx t)
{
    using solver::Fractional;
    const auto zero = Fractional{0, 1};
    const auto one = Fractional{1, 1};
    const auto n = g.nodes;
    const auto m = std::accumulate(g.edges.begin(), g.edges.end(), 0u,
                                   [](const auto acc, const auto& e) { return acc + e.size(); });

    std::vector<Fractional> obj(n, zero);
    obj[t] = one;  // tに依存
    std::vector<Fractional> con(m + 1, zero);
    // TODO: m * nが32bitの範囲に収まりきらなかったらどうしよう
    std::vector<Fractional> mat((m + 1) * n, zero);
    std::size_t edge_cnt = 0;
    for (NodeIdx idx = 0; idx < n; ++idx) {
        for (const auto& [idx2, dist] : g.edges[idx]) {
            con[edge_cnt] = make_int2(dist, 1);
            mat[edge_cnt * n + idx] = solver::negate(one);
            mat[edge_cnt * n + idx2] = one;
            edge_cnt++;
        }
    }
    con[m] = zero;
    mat[m * n + s] = one;  // sに依存
    return solver::LPSolver(n, m + 1, mat.data(), obj.data(), con.data(), zero);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "num of arguments is smaller than 3" << std::endl;
        exit(1);
    }
    const char* filepath = argv[1];
    const auto count = atoi(argv[2]);
    int size = std::numeric_limits<int>::max();
    if (argc >= 4) size = atoi(argv[3]);

    const auto g = loadSGBWords(filepath, size);

    std::cout << "loaded nodes: " << g.nodes << std::endl;
    std::cout << "loaded edges: "
              << std::accumulate(g.edges.begin(), g.edges.end(), 0u,
                                 [](const auto acc, const auto& e) { return acc + e.size(); })
              << std::endl;

    std::cout << "Start Dijkstra" << std::endl;
    std::vector<int> answers;
    answers.reserve(count);
    for (int i = 0; i < count; i++) {
        const auto from = (i * 128) % g.nodes;
        const auto to = 1;
        const auto [distance, route] = dijkstra(g, from, to);
        std::cout << "Distance: " << distance << std::endl;
        answers.emplace_back(distance);

        for (std::size_t i = 0; i < route.size(); ++i) {
            std::cerr << g.idx2id[route[i]];
            if (i + 1 == route.size())
                std::cout << std::endl;
            else
                std::cout << "-->";
        }
    }
    std::cerr << "Finish Dijkstra" << std::endl;

    std::cerr << "Start Simplex" << std::endl;
    using Time = std::chrono::duration<float, std::milli>;
    for (int i = 0; i < count; i++) {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto from = (i * 128) % g.nodes;
        const auto to = 1;
        auto solver = minimumDistance(g, from, to);
        const auto middle = std::chrono::high_resolution_clock::now();
        const auto construct = std::chrono::duration_cast<Time>(middle - start);
        std::cerr << "Construct: " << construct.count() << " [msec]" << std::endl;
        solver.solve(std::chrono::milliseconds(100'000));
        const auto end = std::chrono::high_resolution_clock::now();
        const auto solve = std::chrono::duration_cast<Time>(end - middle);
        std::cerr << "Solve    : " << solve.count() << " [msec]" << std::endl;
        std::cerr << "Distance : " << solver.value().x << std::endl;

        // check answer
        const auto value = solver.value();
        if (answers[i] != value.x) std::cerr << "ERROR!!!!!!" << std::endl;
    }
    std::cerr << "Finish Simplex Method" << std::endl;
}
