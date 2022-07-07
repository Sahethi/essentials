//pragma once is used to reduce the build time as the compiler won’t open and read the file again after the first #include 
#pragma once
 
//it’s just a base header file used by gunrock when defining a new graph primitive 
#include <gunrock/algorithms/algorithms.hxx>
 
namespace gunrock {
namespace greedy_search {
 
struct points_t{
    float x, y;
    float distance;
};

bool comparison(points_t a, points_t b){
    return (a.distance < b.distance);
}

template <typename graph_t, typename vertex_t = typename graph_t::vertex_type> 
__device__ __host__ void greedySearch(graph_t const& G,
                                       points_t const* points,
                                       int n, int k, points_t xq
                                       vertex_t const& v) {
  // Calculate distance

    auto start_edge = G.get_starting_edge(v);
    auto num_neighbors = G.get_number_of_neighbors(v);

    for (auto e = start_edge; e < start_edge + num_neighbors; e++) {
    vertex_t u = G.get_destination_vertex(e);
    points[u].distance = (points[u].x - xq.x) * (points[u].x - xq.x) + (points[u].y - xq.y) * (points[u].y - xq.y);
    }

    std::sort(points, points+n, comparison);

    for (auto e = start_edge; e < start_edge + num_neighbors; e++) {
    vertex_t u = G.get_destination_vertex(e);
    cout << "x = " << points[u].x ;
    cout << " ; y = " << points[u].y << "\n";
    }

}

struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

struct result_t {
  points_t* points;
  result_t(points_t* _points) : points(_points) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> visited;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  void init() override {
    auto g = this->get_graph();
    auto n_edges = g.get_number_of_edges();
  }

  void reset() override {
  }
};


template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto visited = P->visited.data().get();

    auto iteration = this->iteration;

    auto search = [distances, single_source, iteration] __host__ __device__(
                      vertex_t const& source,    // ... source
                      vertex_t const& neighbor,  // neighbor
                      edge_t const& edge,        // edge
                      weight_t const& weight     // weight (tuple).
                      ) -> bool {

      // Simpler logic for the above.
      auto old_distance =
          math::atomic::min(&distances[neighbor], iteration + 1);
      return (iteration + 1 < old_distance);
    };

    auto remove_invalids =
        [] __host__ __device__(vertex_t const& vertex) -> bool {
      return true;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, search, context);

    // Execute filter operator to remove the invalids.
    // @todo: Add CLI option to enable or disable this.
    // operators::filter::execute<operators::filter_algorithm_t::compact>(
    // G, E, remove_invalids, context);
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          points_t* points,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t<vertex_t>;
  using result_type = result_t;

  param_type param(single_source);
  result_type result(points);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace greedy_search
}  // namespace gunrock
