#include <gunrock/algorithms/nn.hxx>
#include <fstream>

using namespace gunrock;
using namespace memory;
using namespace std;
void test_sssp(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types
  int k = 3; 

  using vertex_t = int;
  using edge_t = int;
  using weight_t = float;

  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;

  // --
  // IO

  csr_t csr;
  std::string filename = argument_array[1];

  if (util::is_market(filename)) {
    io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
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
  vertex_t single_source = 0;  // rand() % n_vertices;
  vertex_t query_point[2] = {10, 4};
  std::cout << "Single Source = " << single_source << std::endl;

   int full_vectors[n_vertices][2];

  //storing graph in a custom template type
  ifstream infile("/content/essentials/examples/algorithms/nn/points.txt");
  string line;
  int i = 0;
  while (getline(infile, line)) {
      istringstream iss(line);
      int a, b;
      if (!(iss >> a >> b)) { break; } // error
      full_vectors[i][0] = a; 
      full_vectors[i][1] = b;
      i++;
  }

  // Checking if it works
  // for(i=0;i<n_vertices;i++){    
  //   cout<<g[i].node_id<<endl;
  //   cout<<g[i].points[0]<<endl;
  //   cout<<g[i].points[1]<<endl;
  // }    

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
  thrust::device_vector<vertex_t> top_k(n_vertices);

  gunrock::nn::run(G, single_source, distances.data().get(), full_vectors, query_point, k, top_k.data().get());

  print::head(distances, 40, "GPU distances");

  //std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
