from .naive import naive_matrix_multiplication as normal_method
from .strassen import strassen_matrix_multiply as strassen_method
from .block_mult import block_matrix_mult as block_method
from .numpy_imp import numpy_mat_mult as numpy_method
from .parallel_cpu_row import parallel_matrix_multiply as parallel_method
from .gpu_accel import gpu_mat_mul as gpu_cupy_method

__all__ = [
    normal_method,
    strassen_method,
    block_method,
    numpy_method,
    parallel_method,
    gpu_cupy_method
]