/**
 * @file bfs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Breadth-First Search algorithm.
 * @version 0.1
 * @date 2020-11-23
 *
 * @copyright Copyright (c) 2020
 *
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <bits/stdc++.h>

using namespace std; 
namespace gunrock {
namespace sample_bfs {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  int k;
  vertex_t* query_point;
  int* full_vectors;
  param_t(vertex_t _single_source, vertex_t* _query_point, int _k, int* _full_vectors) : single_source(_single_source), query_point(_query_point), k(_k) , full_vectors(_full_vectors){}
};

template <typename vertex_t>
struct result_t {
  vertex_t* distances;
  vertex_t* predecessors; /// @todo: implement this.
  vertex_t* top_k;
  result_t(vertex_t* _distances, vertex_t* _predecessors, vertex_t* _top_k, vertex_t n_vertices)
      : distances(_distances), predecessors(_predecessors), top_k(_top_k){}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> visited;  /// @todo not used.

  void init() override {}

  void reset() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<vertex_t>::max());
    thrust::fill(thrust::device, d_distances + this->param.single_source,
                 d_distances + this->param.single_source + 1, 0);
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

    auto query_point = P->param.query_point;
    auto k = P->param.k;
    auto full_vectors = P->param.full_vectors;
    auto top_k = P->result.top_k;

    cout<<iteration<<endl;
    this->active_frontier->print();

    auto search = [distances, single_source, iteration, full_vectors, top_k, query_point, k] __host__ __device__(
                      vertex_t const& source,    // ... source
                      vertex_t const& neighbor,  // neighbor
                      edge_t const& edge,        // edge
                      weight_t const& weight     // weight (tuple).
                      ) -> bool {
      // If the neighbor is not visited, update the distance. Returning false
      // here means that the neighbor is not added to the output frontier, and
      // instead an invalid vertex is added in its place. These invalides (-1 in
      // most cases) can be removed using a filter operator or uniquify.
      if (distances[neighbor] != std::numeric_limits<vertex_t>::max())
        return false;
      else
        return (math::atomic::cas(
                    &distances[neighbor],
                    std::numeric_limits<vertex_t>::max(), iteration + 1) ==
                    std::numeric_limits<vertex_t>::max());

      // Simpler logic for the above.
      // auto old_distance =
      //     math::atomic::min(&distances[neighbor], iteration + 1);
      // return (iteration + 1 < old_distance);
    };

    auto remove_invalids =
        [] __host__ __device__(vertex_t const& vertex) -> bool {
      // Returning true here means that we keep all the valid vertices.
      // Internally, filter will automatically remove invalids and will never
      // pass them to this lambda function.
      return true;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, search, context);

    // Execute filter operator to remove the invalids.
    // @todo: Add CLI option to enable or disable this.
    operators::filter::execute<operators::filter_algorithm_t::compact>(
    G, E, remove_invalids, context);
  }

};  // struct enactor_t

/**
 * @brief Run Breadth-First Search algorithm on a given graph, G, starting from
 * the source node, single_source. The resulting distances are stored in the
 * distances pointer. All data must be allocated by the user, on the device
 * (GPU) and passed in to this function.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param single_source A vertex in the graph (integral type).
 * @param distances Pointer to the distances array of size number of vertices.
 * @param predecessors Pointer to the predecessors array of size number of
 * vertices. (optional, wip)
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::vertex_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          int *full_vectors,
          typename graph_t::vertex_type query_point[],
          int k,
          typename graph_t::vertex_type* top_k,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  //testing
  for(int j=0; j<G.get_number_of_vertices()*2; j+=2){
    cout<<full_vectors[j]<<" ";
    cout<<full_vectors[j+1]<<" ";
  }

  cout<<endl<<k<<endl;
  cout<<query_point[0]<<query_point[1]<<endl;

  param_type param(single_source, query_point, k, full_vectors);
  result_type result(distances, predecessors, top_k, G.get_number_of_vertices());

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace sample_bfs
}  // namespace gunrock