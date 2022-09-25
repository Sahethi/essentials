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
#include <bits/stdc++.h>

using namespace std; 
namespace gunrock {
namespace nn {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  int k;
  vertex_t* query_point;
  int* full_vectors;
  param_t(vertex_t _single_source, vertex_t* _query_point, int _k, int* _full_vectors) : single_source(_single_source), query_point(_query_point), k(_k) , full_vectors(_full_vectors){}
};

template <typename vertex_t, typename weight_t>
struct result_t {
  weight_t* distances;
  vertex_t* top_k;
  result_t(weight_t* _distances, vertex_t* _top_k, vertex_t n_vertices)
      : distances(_distances), top_k(_top_k){}
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

  thrust::device_vector<vertex_t> visited;
  priority_queue <int> top_k_queue;
  priority_queue <int, vector<int>, greater<int> > q;


  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    //Setting the size of visited ('thrust' function)
    visited.resize(n_vertices);

    // Execution policy for a given context (using single-gpu).
    auto policy = this->context->get_context(0)->execution_policy();
    thrust::fill(policy, visited.begin(), visited.end(), -1);
  }

  // The reset method should be called if you want to run the same application multiple times on the same dataset (eg, with different parameters).
  void reset() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();

    auto context = this->get_single_context();
    auto policy = context->execution_policy();

    auto single_source = this->param.single_source;
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    
    //distances should be infinite for non source nodes
    thrust::fill(policy, d_distances + 0, d_distances + n_vertices,
                 std::numeric_limits<weight_t>::max());

    //distance of source node is o
    thrust::fill(policy, d_distances + single_source,
                 d_distances + single_source + 1, 0);

    thrust::fill(policy, visited.begin(), visited.end(),
                 -1);  // This does need to be reset in between runs though

    // while (!q.empty()){
    //     q.pop();
    // }

    // while(!top_k_queue.empty()){
    //   top_k_queue.pop();
    // }

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

  //this method contains the core computational logic for your application
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

    auto search = [distances, single_source, iteration] __host__ __device__(
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
    
  //   // auto find_top_k = [top_k, single_source, k, query_point]
  //   auto shortest_path = [iteration, full_vectors, top_k, query_point, distances, single_source] __host__ __device__(
  //                            vertex_t const& source,    // ... source
  //                            vertex_t const& neighbor,  // neighbor
  //                            edge_t const& edge,        // edge
  //                            weight_t const& weight     // weight (tuple).
  //                            ) -> bool {
  //     weight_t source_distance = thread::load(&distances[source]);
  //     weight_t distance_to_neighbor = source_distance + weight;
      
  //     // Store min(distances[neighbor], new_dist) in distances[neighbor]
  //     // Check if the destination node has been claimed as someone's child
  //     weight_t recover_distance =
  //         math::atomic::min(&(distances[neighbor]), distance_to_neighbor);
      
  //     // if(iteration == 1){
  //     //   for(int j=0; j<8; j++){
  //     //     top_k[0] +=1;
  //     //   }
  //     // }

  //     return (distance_to_neighbor < recover_distance);
  //   };

  //   auto remove_completed_paths = [G, visited, iteration] __host__ __device__(
  //                                     vertex_t const& vertex) -> bool {
  //     if (visited[vertex] == iteration)
  //       return false;

  //     visited[vertex] = iteration;
  //     /// @todo Confirm we do not need the following for bug
  //     /// https://github.com/gunrock/essentials/issues/9 anymore.
  //     // return G.get_number_of_neighbors(vertex) > 0;
  //     return true;
  //   };

  //   // Execute advance operator on the provided lambda
  //   operators::advance::execute<operators::load_balance_t::block_mapped>(
  //       G, E, shortest_path, context);

  //   // Execute filter operator on the provided lambda
  //   operators::filter::execute<operators::filter_algorithm_t::bypass>(
  //       G, E, remove_completed_paths, context);

  //   /// @brief Execute uniquify operator to deduplicate the frontier
  //   /// @note Not required.
  //   // // bool best_effort_uniquification = true;
  //   // // operators::uniquify::execute<operators::uniquify_algorithm_t::unique>(
  //   // // E, context, best_effort_uniquification);
  
  //   auto in_frontier = &(this->frontiers[0]);
  //   auto out_frontier = &(this->frontiers[1]);
  //   //out_frontier->print();
  // }
  
  //convergence condition
  // bool is_converged(gcuda::multi_context_t& context) {
  //   return this->active_frontier->is_empty();
  // }

};  // struct enactor_t


template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::weight_type* distances,      // Output
          int *full_vectors,
          typename graph_t::vertex_type query_point[],
          int k,
          typename graph_t::vertex_type* top_k,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {

  // <user-defined>
  using vertex_t = typename graph_t::vertex_type;
  using weight_t = typename graph_t::weight_type;

  //testing
  // for(int j=0; j<G.get_number_of_vertices()*2; j+=2){
  //   cout<<full_vectors[j]<<" ";
  //   cout<<full_vectors[j+1]<<" ";
  // }

  // Checking if it works
  // for(int i=0;i<14;i++){    
  //   cout<<g[i].node_id<<endl;
  //   cout<<g[i].points[0]<<" "<<g[i].points[1]<<endl;
  // } 
  //cout<<k<<endl;
  //cout<<query_point[0]<<query_point[1]<<endl;

  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t, weight_t>;

  param_type param(single_source, query_point, k, full_vectors);
  result_type result(distances, top_k, G.get_number_of_vertices());
  // </user-defined>

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace nn
}  // namespace gunrock