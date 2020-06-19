from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.__config__ import parallel_info

def parallel_backend():
    parallel_info_string = parallel_info()
    parallel_info_array =  parallel_info_string.splitlines()
    backend_lines = [ line for line in parallel_info_array if line.startswith('ATen parallel backend:')]
    if len(backend_lines) is not 1:
        return None
    backend = backend_lines[0].rsplit(': ')[1]
    return backend

def CppParallelExtension(name, sources, *args,**kwargs):
    parallel_extra_compile_args = []

    backend = parallel_backend()

    if (backend == 'OpenMP'):
        parallel_extra_compile_args = ['-DAT_PARALLEL_OPENMP', '-fopenmp']
    elif (backend == 'native thread pool'):
        parallel_extra_compile_args = ['-DAT_PARALLEL_NATIVE']
    elif (backend == 'native thread pool and TBB'):
         parallel_extra_compile_args = ['-DAT_PARALLEL_NATIVE_TBB']
         
    extra_compile_args = kwargs.get('extra_compile_args', [])
    extra_compile_args += parallel_extra_compile_args
    kwargs['extra_compile_args'] = extra_compile_args

    return CppExtension(name, sources, *args, **kwargs)

setup(
    name='parallel_extension_cpp',
    ext_modules=[
        CppParallelExtension('parallel_extension_cpp',
                     sources = ['parallel_extension.cpp'])
    ] ,
    cmdclass={
        'build_ext': BuildExtension
    })
