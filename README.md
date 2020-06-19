# PyTorch custom operator experiment using at::parallel_for 

Inspired by a post on discuss.pytorch.org [Using at::parallel_for in a custom operator](https://discuss.pytorch.org/t/using-at-parallel-for-in-a-custom-operator/82747)

Using at::parallel_for in custom operators requires the correct compiler arguments for the maching parallel backend.

However torch.utils.cpp_extension does not currently implement this functionality.
This directory is a proof of concept hack to extract the correct backend from torch.\__config\__.parallel_info.
A wrapper around CppExtension (CppParallelExtension) is then used to  extract the parallel backend from torch and add the required compiler arguments.

__!!! Only tested using the docker image produced using pytorch/Dockerfile !!!__
__!!! Proof of concept code (No error handling ... use on your own risk !!!__



The extension build in this test simply allocates a vector of a specified size and then utilizes at::parallel_for(0, size, 1, parallel_for_body). The parallel_for__body sets the first entry in the range to one and the rest to zero. A sum over the vector is then used to count the number of times the parallel_for__body was launched (hopefully in parallel)

Usage Example:

```
>>> import torch
>>> import parallel_extension_cpp
>>> parallel_extension_cpp.concurrency_test(21)
11
>>> parallel_extension_cpp.concurrency_test(128)
16
>>> parallel_extension_cpp.concurrency_test(99)
15
>>> parallel_extension_cpp.concurrency_test(150)
15
>>> parallel_extension_cpp.concurrency_test(151)
16
```