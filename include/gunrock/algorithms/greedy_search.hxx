//pragma once is used to reduce the build time as the compiler won’t open and read the file again after the first #include 
#pragma once
 
//it’s just a base header file used by gunrock when defining a new graph primitive 
#include <gunrock/algorithms/algorithms.hxx>
 
namespace gunrock {
namespace greedy_search {
 
//defining the template to specify our graph parameters i.e. our variable. Typename is a like datatype that is predefined in the gunrock library, here we use vertex_t
template <typename vertex_t>
struct param_t {
//single source being the source node
 vertex_t single_source;
 param_t(vertex_t _single_source) : single_source(_single_source) {}
};
 
//distances and predecessors are our two variables that we would need for our bfs algorithm
template <typename vertex_t>
struct result_t {
//resulting distances are stored in distances
 vertex_t* distances;
 
//Pointer to the predecessors array of size number of vertices. 
 vertex_t* predecessors;
 result_t(vertex_t* _distances, vertex_t* _predecessors)
     : distances(_distances), predecessors(_predecessors) {}
};
 
//here our template is taking graph, param and result, where graph is our input and problem defined our BFS problem
template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
 param_type param;
 result_type result;
 
//we pass the following arguments in our problem 
 problem_t(graph_t& G,
           param_type& _param,
           result_type& _result,
           std::shared_ptr<gcuda::multi_context_t> _context)
     : gunrock::problem_t<graph_t>(G, _context),
       param(_param),
       result(_result) {}
 
//defining the variables for the graph
 using vertex_t = typename graph_t::vertex_type;
 using edge_t = typename graph_t::edge_type;
 using weight_t = typename graph_t::weight_type;
 
//thrust is a c++ library which allows you to perform  high performance parallel applications and is based on STL
//just defining the visited array incase we need it
 thrust::device_vector<vertex_t> visited;
 
//initialization function
 void init() override {}
 
//you reset the values of your variables in this function
 void reset() override {
   auto n_vertices = this->get_graph().get_number_of_vertices();
   auto d_distances = thrust::device_pointer_cast(this->result.distances);
//thrust::device_point_cast is a raw pointer which is presumed to point to a location in device memory.
 
//thrust::fill is used to specify value to every element in the range [first, last, value]
   thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices,
                std::numeric_limits<vertex_t>::max());
   thrust::fill(thrust::device, d_distances + this->param.single_source,
                d_distances + this->param.single_source + 1, 0);
 }
};
 
//now we are implementing our enactor part
template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
 enactor_t(problem_t* _problem,
           std::shared_ptr<gcuda::multi_context_t> _context)
     : gunrock::enactor_t<problem_t>(_problem, _context) {}
 
 using vertex_t = typename problem_t::vertex_t;
 using edge_t = typename problem_t::edge_t;
 using weight_t = typename problem_t::weight_t;
 using frontier_t = typename enactor_t<problem_t>::frontier_t;
 
//preparing the frontier array 
 void prepare_frontier(frontier_t* f,
                       gcuda::multi_context_t& context) override {
   auto P = this->get_problem();
   f->push_back(P->param.single_source);
 }
 
//loop is defined for every algorithm 
 void loop(gcuda::multi_context_t& context) override {
//defining a DataSlice here where we have Problem, Enactor and our graph
   auto E = this->get_enactor();
   auto P = this->get_problem();
   auto G = P->get_graph();
 
//single source being where our graph starts from, hence storing the data into the respective variables from the problem 
   auto single_source = P->param.single_source;
   auto distances = P->result.distances;
   auto visited = P->visited.data().get();
 
   auto iteration = this->iteration;
 
//this is our breadth first search logic
//basically is the neighbor is not visited then update the distance then one will return false. 
//Returning false means that the neighbor is not added to the output frontier and instead invalid is added in this case. 
//invalid (sometimes -1) can be removed using filter operate
   auto search = [distances, single_source, iteration] __host__ __device__(
                     vertex_t const& source,  // source
                     vertex_t const& neighbor,  // neighbor
                     edge_t const& edge,        // edge
                     weight_t const& weight     // weight
                     ) -> bool {
     auto old_distance =
         math::atomic::min(&distances[neighbor], iteration + 1);
     return (iteration + 1 < old_distance);
   };
 
// Returning true here means that we keep all the valid vertices.
// filter operator will remove invalids
 
   auto remove_invalids =
       [] __host__ __device__(vertex_t const& vertex) -> bool {
     return true;
   };
 
   // Execute advance operator on the provided lambda
operators::advance::execute<operators::load_balance_t::block_mapped>(
       G, E, search, context);
 
//Remove all the invalids
operators::filter::execute<operators::filter_algorithm_t::compact>(
   G, E, remove_invalids, context);
 }
 
}; 
 
//run will allow you to calculate the time taken to run the algorithm on device i.e. GPU
 
template <typename graph_t>
float run(graph_t& G,
         typename graph_t::vertex_type& single_source,  
// Parameter
         typename graph_t::vertex_type* distances,      
// Output
         typename graph_t::vertex_type* predecessors,   
// Output
         std::shared_ptr<gcuda::multi_context_t> context =
             std::shared_ptr<gcuda::multi_context_t>(
                 new gcuda::multi_context_t(0))  // Context
) {
 using vertex_t = typename graph_t::vertex_type;
 using param_type = param_t<vertex_t>;
 using result_type = result_t<vertex_t>;
 
 param_type param(single_source);
 result_type result(distances, predecessors);
 
 using problem_type = problem_t<graph_t, param_type, result_type>;
 using enactor_type = enactor_t<problem_type>;
 
 problem_type problem(G, param, result, context);
 problem.init();
 problem.reset();
 
//giving a call to enact function
 enactor_type enactor(&problem, context);
 return enactor.enact();
}
} 
}
