#include <gunrock/algorithms/greedy_search.hxx>
#include "greedy_search_cpu.hxx"  // Reference implementation
#include <fstream>

using namespace std;
using namespace gunrock;
using namespace memory;

vector<double> calculate_distances(int x, int y){
  ifstream infile("/content/essentials/examples/algorithms/greedy_search/points.txt");
  string line;
  vector<double> euclidean;
  while (getline(infile, line)) {
      istringstream iss(line);
      int a, b;
      if (!(iss >> a >> b)) { break; } // error
      double dist = sqrt(pow(x-a, 2) + pow(y-b, 2));
      euclidean.push_back(dist);
  }

  return euclidean;
}

void test_greedy_search(int num_arguments, char** argument_array, vector<double> euclidean_distances) {
  if (num_arguments != 4) {
     cerr << "usage: ./bin/<program-name> filename.mtx x y" <<  endl;
    exit(1);
  }

  // --
  // Define types

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  csr_t csr;
   string filename = argument_array[1];

  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
     cerr << "Unknown file format: " << filename <<  endl;
    exit(1);
  }

  // --
  // Build graph

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows,               // rows
      csr.number_of_columns,            // columns
      csr.number_of_nonzeros,           // nonzeros
      csr.row_offsets.data().get(),     // row_offsets
      csr.column_indices.data().get(),  // column_indices
      csr.nonzero_values.data().get()   // values
  );  // supports row_indices and column_offsets (default = nullptr)

  // --
  // Params and memory allocation
  srand(time(NULL));
  vertex_t n_vertices = G.get_number_of_vertices();
  thrust::device_vector<vertex_t> nodes(n_vertices);
  vertex_t single_source = 0;
  for(int j = 0; j < n_vertices; j++){
    nodes[j] = j;
  }
  single_source = 0;
  cout << "Single Source = " << single_source <<  endl;
    // --
    // GPU Run

    /// An example of how one can use std::shared_ptr to allocate memory on the
    /// GPU, using a custom deleter that automatically handles deletion of the
    /// memory.
    // std::shared_ptr<weight_t> distances(
    //     allocate<weight_t>(n_vertices * sizeof(weight_t)),
    //     deleter_t<weight_t>());
    // std::shared_ptr<vertex_t> predecessors(
    //     allocate<vertex_t>(n_vertices * sizeof(vertex_t)),
    //     deleter_t<vertex_t>());

    thrust::device_vector<weight_t> distances(n_vertices);
    thrust::device_vector<vertex_t> predecessors(n_vertices);

    float gpu_elapsed = 0.0f;
    int num_runs = 5;

    for (auto i = 0; i < num_runs; i++)
      gpu_elapsed += gunrock::greedy_search::run(G, single_source, euclidean_distances, distances.data().get(),
                                        predecessors.data().get(), nodes.data().get());

    gpu_elapsed /= num_runs;

    // --
    // CPU Run

    thrust::host_vector<weight_t> h_distances(n_vertices);
    thrust::host_vector<vertex_t> h_predecessors(n_vertices);
    thrust::host_vector<vertex_t> h_nodes(n_vertices);

    float cpu_elapsed = greedy_search_cpu::run<csr_t, vertex_t, edge_t, weight_t>(
        csr, single_source, euclidean_distances, h_distances.data(), h_predecessors.data(), h_nodes.data());

    int n_errors =
        util::compare(distances.data().get(), h_distances.data(), n_vertices);

    // --
    // Log + Validate


    cout << "\nGPU Elapsed Time : " << gpu_elapsed << " (ms)" <<  endl;
    cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" <<  endl;
    cout << "Number of errors : " << n_errors <<  endl;
  
}

int main(int argc, char** argv) {
  // int x = 10, y = 4;
  vector<double> euclidean;
  int x = atoi(argv[2]);
  int y = atoi(argv[3]);
  euclidean = calculate_distances(10, 4);
  for (auto i = euclidean.begin(); i != euclidean.end(); ++i)
    cout << *i << " ";
  cout<<"\n";
  test_greedy_search(argc, argv, euclidean);
}
