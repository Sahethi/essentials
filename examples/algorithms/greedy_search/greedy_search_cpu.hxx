#pragma once

#include <chrono>
#include <vector>
#include <queue>
#include <iostream>

#include <thrust/host_vector.h>

namespace greedy_search_cpu {

using namespace std;
using namespace std::chrono;

int findIndex(std::vector<double> arr, double item) {
    for (auto i = 0; i < arr.size(); ++i) {
        if (arr[i] == item)
            return i;
    }
    return -1;
}

template <typename vertex_t, typename weight_t>
class prioritize {
 public:
  bool operator()(pair<vertex_t, weight_t>& p1, pair<vertex_t, weight_t>& p2) {
    return p1.second > p2.second;
  }
};

template <typename csr_t, typename vertex_t, typename edge_t, typename weight_t>

float run(csr_t& csr,
          vertex_t& single_source,
          std::vector<double> euclidean_distances,
          weight_t* distances,
          vertex_t* predecessors, vertex_t* nodes) {
  auto t_start = high_resolution_clock::now();
  priority_queue<double,  vector<double>, greater<double>> q, topk;

 q.push(euclidean_distances[0]);
 while(!q.empty()){
    
    auto pos = findIndex(euclidean_distances, q.top());
    
    thrust::host_vector<edge_t> _row_offsets(
      csr.row_offsets);  // Copy data to CPU
    thrust::host_vector<vertex_t> _column_indices(csr.column_indices);
    thrust::host_vector<weight_t> _nonzero_values(csr.nonzero_values);

    edge_t* row_offsets = _row_offsets.data();
    vertex_t* column_indices = _column_indices.data();
    weight_t* nonzero_values = _nonzero_values.data();

    for (vertex_t i = 0; i < csr.number_of_rows; i++)
      distances[i] = std::numeric_limits<weight_t>::max();


    distances[pos] = 0;

    priority_queue<pair<vertex_t, weight_t>,
                  std::vector<pair<vertex_t, weight_t>>,
                  prioritize<vertex_t, weight_t>>
        pq;
    pq.push(make_pair(pos, 0.0));

    topk.push(euclidean_distances[pos]);
    q.pop();
    while (!pq.empty()) {
      pair<vertex_t, weight_t> curr = pq.top();
      pq.pop();

      vertex_t curr_node = curr.first;
      weight_t curr_dist = curr.second;

      vertex_t start = row_offsets[curr_node];
      vertex_t end = row_offsets[curr_node + 1];
      for (vertex_t offset = start; offset < end; offset++) {
        vertex_t neib = column_indices[offset];
        weight_t new_dist = curr_dist + nonzero_values[offset];
        if (new_dist < distances[neib]) {
          distances[neib] = new_dist;
          pq.push(make_pair(neib, new_dist));
          if(new_dist == 1){
            q.push(euclidean_distances[neib]);
          }
        }
      }
    }
    auto k = 0;
    while(!q.empty() && k == 4){
      k++;
      cout << ' ' <<findIndex(euclidean_distances, q.top()); 
      q.pop();
    }
  }
          
  auto t_stop = high_resolution_clock::now();
  auto elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  return (float)elapsed / 1000;
}

}  // namespace greedy_search_cpu
