#include "solver.h"
#include <iostream>

struct Edge {
    std::size_t from;
    std::size_t to;
    solver::Fractional capacity;
};

struct Graph {
    std::size_t N;  // number of nodes
    std::size_t M;  // number of edges
    std::vector<Edge> edges;
};

solver::LPSolver maximumFlow(const Graph& g, std::size_t source, std::size_t target)
{
    using solver::Fractional;
    const auto zero = make_int2(0, 1);
    const auto one = make_int2(1, 1);
    const auto minus = solver::negate(one);

    std::size_t terms = g.M;
    std::size_t constraints = g.M + 2 * (g.N - 2);
    std::vector<Fractional> mat(terms * constraints, zero);
    std::vector<Fractional> obj(terms, zero);
    std::vector<Fractional> con(constraints, zero);
    // set obj
    for (std::size_t i = 0; i < g.M; ++i) {
        const auto& [f, t, c] = g.edges[i];
        if (f == source) obj[i] = one;
        if (t == source) obj[i] = minus;
    }
    // set con
    for (std::size_t i = 0; i < g.M; ++i) {
        const auto& [f, t, c] = g.edges[i];
        con[i] = c;
    }
    // set mat
    const auto make_idx = [&g, target, source](auto node) {
        if (source < node) node--;
        if (target < node) node--;
        return g.M + 2 * node;
    };
    for (std::size_t i = 0; i < g.M; ++i) {
        mat[i * terms + i] = one;
        const auto& [f, t, c] = g.edges[i];

        // flow conservation at node: f
        if (f != source && f != target) {
            const auto row_f = make_idx(f);
            mat[row_f * terms + i] = one;
            mat[(row_f + 1) * terms + i] = minus;
        }
        // flow conservation at node: t
        if (t != source && t != target) {
            const auto row_t = make_idx(t);
            mat[row_t * terms + i] = minus;
            mat[(row_t + 1) * terms + i] = one;
        }
    }
    return solver::LPSolver(terms, constraints, mat.data(), obj.data(), con.data(), zero);
}

int main()
{
    Graph g = {4,
               5,
               {{0, 1, make_int2(3, 1)},
                {0, 2, make_int2(4, 1)},
                {1, 2, make_int2(2, 1)},
                {1, 3, make_int2(1, 1)},
                {2, 3, make_int2(2, 1)}}};
    auto solver = maximumFlow(g, 0, 3);

    solver.solve(std::chrono::milliseconds(1000));

    const auto optimal = solver.optimal();
    const auto value = solver.value();
    std::cerr << "Optimal?: " << optimal << std::endl;
    std::cerr << "Value   : " << value.x << "/" << value.y << std::endl;
    const auto ptr = solver.solution();
    std::cerr << "Solution: ";
    for (std::int64_t i = 0; i < 5; ++i) std::cerr << ptr[i].x << "/" << ptr[i].y << ", ";
    std::cerr << std::endl;
    if (value.x != 3) std::cerr << "ERROR!!!" << std::endl;
    return 0;
}
