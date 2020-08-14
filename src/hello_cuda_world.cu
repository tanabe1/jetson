#include <cinttypes>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32
#define GPU_NITER 100

#define MAT_SIZE_X 10000
#define MAT_SIZE_Y 10000

// CUDA runtime のエラー処理をラップするマクロ
#define CHECK(func)                                    \
{                                                      \
    const cudaError_t error = func;                    \
    if (error != cudaSuccess)                          \
    {                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);  \
        printf("Code:%d, Reason: %s\n", error,         \
                cudaGetErrorString(error));            \
        cudaDeviceReset();                             \
        exit(EXIT_FAILURE);                            \
    }                                                  \
}

double cpu_second(void) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

double gflops(double sec, uint32_t mat_size_x, uint32_t mat_size_y) {
    double operations = mat_size_x * mat_size_y;
    double gflops = operations * 1.0e-9f / sec;
    return gflops;
}

// CUDAで行列の足し算を実行する関数(kernel)
__global__ void add_matrix_gpu(float *dMat_A, float *dMat_B, float *dMat_G, uint32_t mat_size_x, uint32_t mat_size_y) {
    uint32_t mat_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t mat_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (mat_x >= mat_size_x) {
        return;
    }
    if (mat_y >= mat_size_y) {
        return;
    }

    uint32_t index = mat_y * mat_size_x + mat_x;

    dMat_G[index] = dMat_A[index] + dMat_B[index];
}

// GPUで計算するためのホスト側の処理
void invoke_gpu(const float* __restrict__ hMat_A, const float* __restrict__ hMat_B, float *hMat_G, uint32_t mat_size_x, uint32_t mat_size_y) {
    float *dMat_A = NULL;
    float *dMat_B = NULL;
    float *dMat_G = NULL;
    int nBytes = sizeof(float) * mat_size_x * mat_size_y;
    
    // GPU(device)側にメモリを確保
    CHECK(cudaMalloc((float **) &dMat_A, nBytes));
    CHECK(cudaMalloc((float **) &dMat_B, nBytes));
    CHECK(cudaMalloc((float **) &dMat_G, nBytes));

    // GPU(device)側にホスト側のメモリの内容をコピーする
    CHECK(cudaMemcpy(dMat_A, hMat_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dMat_B, hMat_B, nBytes, cudaMemcpyHostToDevice));

    // 計算処理を分割する gird / block を計算する
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((mat_size_x + block.x - 1) / block.x, (mat_size_y + block.y - 1) / block.y);
    printf("Grid = (%d, %d), Block = (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // GPUで計算処理を行った際のパフォーマンスを計測する
    double start_sec = cpu_second();
    for (int i = 0; i < GPU_NITER; i++) {
        add_matrix_gpu<<<grid, block>>> (dMat_A, dMat_B, dMat_G, mat_size_x, mat_size_y);
    }
    CHECK(cudaDeviceSynchronize());
    double elapsed_sec = (cpu_second() - start_sec) / GPU_NITER;

    // 結果をホスト側にコピーする
    CHECK(cudaMemcpy(hMat_G, dMat_G, nBytes, cudaMemcpyDeviceToHost));

    // 最初の10個だけ結果を表示する
    for (uint32_t i = 0; i < mat_size_x * mat_size_y; i++) {
        if (i < 10) {
            printf("A[%d]=%8.4f, B[%d]=%8.4f, G[%d]=%8.4f\n", i, hMat_A[i], i, hMat_B[i], i, hMat_G[i]);
        }
    }
    printf("GPU: Time elapsed %lf sec (%lf GFLOPS)\n", elapsed_sec, gflops(elapsed_sec, mat_size_x, mat_size_y));

    CHECK(cudaFree(dMat_A));
    CHECK(cudaFree(dMat_B));
    CHECK(cudaFree(dMat_G));
    CHECK(cudaDeviceReset());
}

// CPUで行列の足し算を実行する関数(シングルスレッド)
void add_matrix_cpu(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size_x, uint32_t mat_size_y) {
    for (uint32_t y = 0; y < mat_size_y; y++) {
        for (uint32_t x = 0; x < mat_size_x; x++) {
            uint32_t index = y * mat_size_x + x;
            hMat_C[index] = hMat_A[index] + hMat_B[index];
        }
    }
}

// CPUで計算を実行する
void invoke_cpu(float *hMat_A, float *hMat_B, float *hMat_C, uint32_t mat_size_x, uint32_t mat_size_y) {
    double start_sec = cpu_second();
    for (int i = 0; i < GPU_NITER; i++) {
        add_matrix_cpu(hMat_A, hMat_B, hMat_C, mat_size_x, mat_size_y);
    }
    double elapsed_sec = (cpu_second() - start_sec) / GPU_NITER;

    // 最初の10個だけ結果を表示する
    for (uint32_t i = 0; i < mat_size_x * mat_size_y; i++) {
        if (i < 10) {
            printf("A[%d]=%8.4f, B[%d]=%8.4f, G[%d]=%8.4f\n", i, hMat_A[i], i, hMat_B[i], i, hMat_C[i]);
        }
    }
    printf("CPU: Time elapsed %lf sec (%lf GFLOPS)\n", elapsed_sec, gflops(elapsed_sec, mat_size_x, mat_size_y));
}

int main(void)
{
    uint32_t mat_size_x = MAT_SIZE_X;
    uint32_t mat_size_y = MAT_SIZE_Y;
    int nBytes = sizeof(float) * mat_size_x * mat_size_y;

    float *hMat_A;
    float *hMat_B;
    float *hMat_G;

    // 行列Aと行列B、結果を格納する行列Gのためのメモリを確保する
    hMat_A = (float *) malloc(nBytes);
    hMat_B = (float *) malloc(nBytes);
    hMat_G = (float *) malloc(nBytes);

    // 乱数で行列Aと行列Bを初期化する
    time_t t;
    srand((unsigned int) time(&t));
    for (uint32_t i = 0; i < mat_size_x * mat_size_y; i++) {
        hMat_A[i] = (float)(rand() % 100000) / 10000.0f;
        hMat_B[i] = (float)(rand() % 100000) / 10000.0f;
    }

     
    invoke_gpu(hMat_A, hMat_B, hMat_G, mat_size_x, mat_size_y);
    invoke_cpu(hMat_A, hMat_B, hMat_G, mat_size_x, mat_size_y);

    // 確保したメモリを開放する
    free(hMat_A);
    free(hMat_B);
    free(hMat_G);
}
