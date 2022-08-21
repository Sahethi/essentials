/**
 * @file nn.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Single-Source Shortest Path algorithm.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace nn {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* distances;
  result_t(weight_t* _distances) : distances(_distances) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(
    graph_t& G,
    param_type& _param,
    result_type& _result,
    std::shared_ptr<cuda::multi_context_t> _context
  ) : gunrock::problem_t<graph_t>(G, _context), param(_param), result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

    // Create a datastructure that is internal to the application, and will not be returned to
  // the user.
  thrust::device_vector<vertex_t> visited;

  // `init` function, described above.  This should be called once, when `problem` gets instantiated.
  void init() {
    // Get the graph
    auto g = this->get_graph();
    
    // Get number of vertices from the graph
    auto n_vertices = g.get_number_of_vertices();   
    
    // Set the size of `visited` (`thrust` function)
    visited.resize(n_vertices);
  }

  // `reset` function, described above.  Should be called
  // - after init, when `problem` is instantiated
  // - between subsequent application runs, eg when you change the parameters

  void reset() {
    auto g = this->get_graph();
    
    auto distances  = this->result.distances;
    auto n_vertices = g.get_number_of_vertices();
    
    // fill `distances` with the max `weight_t` value
    // ... because at the beginning of `sssp`, distance to all non-source nodes should be infinity
    thrust::fill(
      thrust::device, 
      distances + 0, 
      distances + n_vertices,
      std::numeric_limits<weight_t>::max()
    );

    // Set the `single_source`'th element of distances to 0
    // ... because at the beginning of `sssp`, distance to the source node should be 0
    thrust::fill(
      thrust::device,
      distances + this->param.single_source,
      distances + this->param.single_source + 1,
      0
    );

    // Fill `visited` with -1 (`thrust` function)
    // ... because at the beginning of `sssp`, no nodes have been
    thrust::fill(thrust::device, visited.begin(), visited.end(), -1);
  }
};


// <boilerplate>
template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;
// </boilerplate>

// How to initialize the frontier at the beginning of the application.
  // In this case, we just need to add a single node
  void prepare_frontier(frontier_t* f, cuda::multi_context_t& context) override {
    // get pointer to the problem
    auto P = this->get_problem();
    
    // add `single_source` to the frontier
    f->push_back(P->param.single_source);
  }


   // One iteration of the application
  void loop(cuda::multi_context_t& context) override {

    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    // Get parameters and datastructures
    // Note that `P->visited` is a thrust vector, so we need to unwrap again
    auto single_source = P->param.single_source;
    auto distances     = P->result.distances;
    auto visited       = P->visited.data().get();

    // Get current iteration of application
    auto iteration = this->iteration;

    // Advance operator for single-source shortest paths application
    auto shortest_path = [distances, single_source] __host__ __device__(
      vertex_t const& source,    // source of edge
      vertex_t const& neighbor,  // destination of edge
      edge_t const& edge,        // id of edge
      weight_t const& weight     // weight of edge
    ) -> bool {
      
      // Get implied distance to neighbor using a path through source
      weight_t new_dist = distances[source] + weight;
      
      // Store min(distances[neighbor], new_dist) in distances[neighbor]
      weight_t old_dist = math::atomic::min(distances + neighbor, new_dist);

      // If the new distance is better than the previously known best_distance, add `neighbor` to 
      // the frontier
      return new_dist < old_dist;
    };

    // Execute advance operator
    // More documentation on these flags is available in:
    // include/gunrock/framework/operators/advance/advance.hxx
    operators::advance::execute<operators::load_balance_t::block_mapped,
                                operators::advance_direction_t::forward,
                                operators::advance_io_type_t::vertices,
                                operators::advance_io_type_t::vertices>(
        G, E, shortest_path, context);
    
    
    auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
      vertex_t const& vertex
    ) -> bool {
      
      // Drop nodes w/ no out-degree, since we can't continue search from them
      if (G.get_number_of_neighbors(vertex) == 0) return false;
      
      // Uniquify the frontier
      if (visited[vertex] == iteration) return false;
      visited[vertex] = iteration;
      
      // Otherwise, keep this node in the frontier
      return true;
    };

    // Execute filter operator
    // @see include/gunrock/framework/operators/filter/filter.hxx
    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, E, remove_completed_paths, context);
  }

  virtual bool is_converged(cuda::multi_context_t& context) {
    return this->active_frontier->is_empty();
  }

};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::weight_type* distances,      // Output
          // Context for application (eg, GPU + CUDA stream it will be executed on)
          std::shared_ptr<cuda::multi_context_t> context =
              std::shared_ptr<cuda::multi_context_t>(
                  new cuda::multi_context_t(0))
) {
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  // instantiate `param` and `result` templates
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  // initialize `param` and `result` w/ the appropriate parameters / data structures
  param_type param(single_source);
  result_type result(distances);

  // <boilerplate> This code probably should be the same across all applications, 
  // unless maybe you're doing something like multi-gpu / concurrent function calls

  // instantiate `problem` and `enactor` templates.
  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  // initialize problem; call `init` and `reset` to prepare data structures
  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // initialize enactor; call enactor, returning GPU elapsed time
  enactor_type enactor(&problem, context);
  return enactor.enact();
  // </boilerplate>
}

}  // namespace nn
}  // namespace gunrock