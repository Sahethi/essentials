#pragma once
//chrono is used to deal with date and time
#include <chrono>
#include <vector>
#include <queue>
 
namespace sample_bfs_cpu {
 
using namespace std;
using namespace std::chrono;
 
template <typename vertex_t>
 
//Here we are defining the way to prioritize for the priority queue
class prioritize {
public:
 bool operator()(pair<vertex_t, vertex_t>& p1, pair<vertex_t, vertex_t>& p2) {
   return p1.second > p2.second;
 }
};
 
//csr is the graph format 
template <typename csr_t, typename vertex_t, typename edge_t>
float run(csr_t& csr,
         vertex_t& single_source,
         vertex_t* distances,
         vertex_t* predecessors) {
 thrust::host_vector<edge_t> _row_offsets(
     csr.row_offsets);  // Copy data to CPU
 thrust::host_vector<vertex_t> _column_indices(csr.column_indices);
 
 edge_t* row_offsets = _row_offsets.data();
 vertex_t* column_indices = _column_indices.data();
 
//traversing the graph
 for (vertex_t i = 0; i < csr.number_of_rows; i++)
   distances[i] = std::numeric_limits<vertex_t>::max();
 
 auto t_start = high_resolution_clock::now();
 
 distances[single_source] = 0;
 
//defining the priority queue
 priority_queue<pair<vertex_t, vertex_t>,
                std::vector<pair<vertex_t, vertex_t>>, prioritize<vertex_t>> pq;
 
 pq.push(make_pair(single_source, 0.0));
 
 while (!pq.empty()) {
   pair<vertex_t, vertex_t> curr = pq.top();
   pq.pop();
 
   vertex_t curr_node = curr.first;
   vertex_t curr_dist = curr.second;
 
   vertex_t start = row_offsets[curr_node];
   vertex_t end = row_offsets[curr_node + 1];
   
   //Our main logic for BFS, where we are going from start vertex to the end 
   for (vertex_t offset = start; offset < end; offset++) {
     //neib is the neighbour
     vertex_t neib = column_indices[offset];
	//increase one hop on visiting the neighbour
     vertex_t new_dist = curr_dist + 1;
     //if new distance if less than the distance that it takes to reach the neighbor then replace the distance
     if (new_dist < distances[neib]) {
       distances[neib] = new_dist;
//make a pair of that neighbor and the number of hops accordingly 
       pq.push(make_pair(neib, new_dist));
     }
   }
 }
 
//Here is measure the total elapsed time on CPU
 auto t_stop = high_resolution_clock::now();
 auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
 return (float)elapsed / 1000;
}
 
}  // namespace sample_bfs_cpu
