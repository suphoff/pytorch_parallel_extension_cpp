#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <ATen/ATen.h>

#include <ATen/Parallel.h>

// Test concurrency using a parallel_for loop
// Creates a vector of size max(test_size,1) and then uses a parallel_for loop
// on the vector that

int64_t concurrency_test(int64_t test_size)
{
  
  int64_t vsize = std::max(test_size,(int64_t) 1);
    
  auto tensor = at::empty({vsize},  at::kLong);
  auto tensor_a = tensor.accessor<int64_t,1>();
  
  at::parallel_for(0, vsize, 1,
		   [&] (int64_t begin, int64_t end) {
		     tensor_a[begin] = 1;
		     for (int64_t b = begin + 1 ; b < end; b++) {
		       tensor_a[b] = 0;
		     }
		   });
  
    
  auto sum_tensor = tensor.sum(0);

  int64_t sum = *sum_tensor.data_ptr<int64_t>();
          
  return sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("concurrency_test",  &concurrency_test, "Estimate Concurrency in at::parallel_for");
}
