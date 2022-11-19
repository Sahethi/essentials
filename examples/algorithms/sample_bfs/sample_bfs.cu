#include <gunrock/algorithms/sample_bfs.hxx>
#include "sample_bfs_cpu.hxx"
#include <fstream>
 
using namespace gunrock;
using namespace memory;
using namespace std;
//test_bfs has all our main functionality where we are creating out executable file in bin folder and getting the output and the statistics. 
void test_bfs(int num_arguments, char** argument_array) {
 if (num_arguments != 2) {
   std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
   exit(1);
 }
 
 //defining types
 int k = 3; 


 using vertex_t = int;
 using edge_t = int;
 using weight_t = float;
 
//defining the graph format
 using csr_t =
     format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
 
 csr_t csr;
 std::string filename = argument_array[1];
 
// checking for different graph formats and this would remain same for any graph primitive
 if (util::is_market(filename)) {
   io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
   csr.from_coo(mm.load(filename));
 } else if (util::is_binary_csr(filename)) {
   csr.read_binary(filename);
 } else {
   std::cerr << "Unknown file format: " << filename << std::endl;
   exit(1);
 }
 
 thrust::device_vector<vertex_t> row_indices(csr.number_of_nonzeros);
 thrust::device_vector<vertex_t> column_indices(csr.number_of_nonzeros);
 thrust::device_vector<edge_t> column_offsets(csr.number_of_columns + 1);
 
//now one would build the graph with the given data
 auto G =
     graph::build::from_csr<memory_space_t::device,
                            graph::view_t::csr>(
         csr.number_of_rows,               // rows
         csr.number_of_columns,            // columns
         csr.number_of_nonzeros,           // nonzeros
         csr.row_offsets.data().get(),     // row_offsets
         csr.column_indices.data().get(), // column_indices
         csr.nonzero_values.data().get(),  // values
         row_indices.data().get(),         // row_indices
         column_offsets.data().get()      // column_offsets
     );
 
 //parameters
 
 vertex_t single_source = 0;
 vertex_t n_vertices = G.get_number_of_vertices();
 vertex_t query_point[2] = {10, 4};
 cout << "Single Source = " << single_source << endl;

 int *full_vectors = (int*)malloc(n_vertices * 2 * sizeof(int));

  //storing graph in a custom template type
  ifstream infile("/content/essentials/examples/algorithms/sample_bfs/points.txt");
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


 thrust::device_vector<vertex_t> distances(n_vertices);
 thrust::device_vector<vertex_t> predecessors(n_vertices);
 //thrust::device_vector<vertex_t> top_k(n_vertices);
 int *top_k = (int*)malloc(n_vertices * sizeof(int));
 
// run problem this is will total time elapsed for GPU
 
 float gpu_elapsed = gunrock::sample_bfs::run(
     G, single_source, distances.data().get(), predecessors.data().get(), full_vectors, query_point, k, top_k);
 
 // run problem this is will total time elapsed for CPU
 thrust::host_vector<vertex_t> h_distances(n_vertices);
 thrust::host_vector<vertex_t> h_predecessors(n_vertices);
 
 float cpu_elapsed = sample_bfs_cpu::run<csr_t, vertex_t, edge_t>(
     csr, single_source, h_distances.data(), h_predecessors.data());
 
 int n_errors =
     util::compare(distances.data().get(), h_distances.data(), n_vertices);
 
 
//Gives distances for the first 40 nodes of the mentioned dataset
 print::head(distances, 40, "GPU distances");
 print::head(h_distances, 40, "CPU Distances");
 
//prints elapsed time for CPU and GPU
 std::cout << "GPU Elapsed Time : " << gpu_elapsed << " (ms)" << std::endl;
 std::cout << "CPU Elapsed Time : " << cpu_elapsed << " (ms)" << std::endl;
 std::cout << "Number of errors : " << n_errors << std::endl;
}
 
//Calling the main function
int main(int argc, char** argv) {
 test_bfs(argc, argv);
}
