import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

from pycuda.compiler import SourceModule

BLOCKSIZE = 32
GPU_NITER = 1

MAT_SIZE_X = 1000
MAT_SIZE_Y = 1000

def gflops(sec, mat_size_x, mat_size_y):
    operations = mat_size_x * mat_size_y
    gflops = operations * 1e-9 / sec
    return gflops

## CUDAカーネルを定義
mod = SourceModule("""
__global__ void add_matrix_gpu(const float* __restrict__ dMat_A, float* __restrict__ dMat_B, float *dMat_G, const int mat_size_x, const int mat_size_y) {
    int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
    int mat_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (mat_x >= mat_size_x) {
        return;
    }
    if (mat_y >= mat_size_y) {
        return;
    }

    const int index = mat_y * mat_size_x + mat_x;

    dMat_G[index] = dMat_A[index] + dMat_B[index];
}
""")


add_matrix_gpu = mod.get_function("add_matrix_gpu")
block = (BLOCKSIZE, BLOCKSIZE, 1)
grid = ((MAT_SIZE_X + block[0] - 1) // block[0], (MAT_SIZE_Y + block[1] - 1) // block[1])
print("Grid = ({0}, {1}), Block = ({2}, {3})".format(grid[0], grid[1], block[0], block[1]))

start = cuda.Event()
end = cuda.Event()

h_a = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_b = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_d = numpy.empty_like(h_a)

start.record()
for i in range(GPU_NITER):
    add_matrix_gpu(cuda.In(h_a), cuda.In(h_b), cuda.Out(h_d), numpy.int32(MAT_SIZE_X), numpy.int32(MAT_SIZE_Y), block = block, grid = grid)
end.record()
end.synchronize()

elapsed_sec = start.time_till(end) * 1e-3 / GPU_NITER

for y in range(MAT_SIZE_Y):
    for x in range(MAT_SIZE_X):
        i = y * MAT_SIZE_X + x
        if i < 10:
            print("A[%d]=%8.4f, B[%d]=%8.4f, D[%d]=%8.4f" % (i, h_a[x][y], i, h_b[x][y], i, h_d[x][y]))
        else:
            break

print("GPU: Time elapsed %f sec (%lf GFLOPS)" % (elapsed_sec, gflops(elapsed_sec, MAT_SIZE_X, MAT_SIZE_Y)))

cuda.Context.synchronize()
