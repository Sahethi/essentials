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

  int *full_vectors = (int*)malloc(n_vertices * 2 * sizeof(int));

  //storing graph in a custom template type
  ifstream infile("/content/essentials/examples/algorithms/nn/points.txt");
  string line;
  int i = 0, j = 0;
  while (getline(infile, line)) {
      istringstream iss(line);
      int a, b;
      if (!(iss >> a >> b)) { break; } // error
      full_vectors[j] = a; 
      full_vectors[j+1] = b;
      i++;
      j+=2;
  }

  thrust::device_vector<weight_t> distances(n_vertices);
  thrust::device_vector<vertex_t> top_k(n_vertices);

  gunrock::nn::run(G, single_source, distances.data().get(), full_vectors, query_point, k, top_k.data().get());

  //Print the first k elements of a vector. Here it is distance.
  print::head(distances, 40, "GPU distances");
}

int main(int argc, char** argv) {
  test_sssp(argc, argv);
}
