#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 定义 BLOCK_NUM 和 THREAD_NUM
#define BLOCK_NUM 4  
#define THREAD_NUM 4

// __global__ void mat_mul(float *mat1, float *mat2, float *result, int M, int K, int N) {
//     const int bid = blockIdx.x;
//     const int tid = threadIdx.x;
//     const int row = bid * THREAD_NUM + tid;

//     if (row < M) {
//         for (int c = 0; c < N; c++) {
//             float sum = 0.0f;
//             for (int n = 0; n < K; n++) {
//                 sum += mat1[row * K + n] * mat2[n * N + c];
//             }
//             result[row * N + c] = sum;
//         }
//     }
// }

#define TILE_SIZE 16

// CUDA kernel for matrix multiplication using shared memory tiling
__global__ void mat_mul(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float C_value = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            C_value += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result to C
    if (row < M && col < N) {
        C[row * N + col] = C_value;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <M> <K> <N>\n", argv[0]);
        return 1;
    }

    // 从命令行参数获取 M, K, N
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    printf("M: %d, K: %d, N: %d\n", M, K, N);

    float *mat1, *mat2, *result;
    float *g_mat1, *g_mat2, *g_mat_result;

    // 分配主机内存
    mat1 = (float*) malloc(M * K * sizeof(float));
    mat2 = (float*) malloc(K * N * sizeof(float));
    result = (float*) malloc(M * N * sizeof(float));

    // 初始化矩阵
    for (int i = 0; i < M * K; i++) {
        mat1[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < K * N; i++) {
        mat2[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < M * N; i++) {
        result[i] = 0.0f;
    }

    // 分配设备内存
    cudaMalloc((void **)&g_mat1, sizeof(float) * M * K);
    cudaMalloc((void **)&g_mat2, sizeof(float) * K * N);
    cudaMalloc((void **)&g_mat_result, sizeof(float) * M * N);

    // 将数据从主机复制到设备
    cudaMemcpy(g_mat1, mat1, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    // 计算矩阵乘法
    dim3 grid((M + THREAD_NUM - 1) / THREAD_NUM); // 计算需要的 block 数量
    dim3 block(THREAD_NUM);
    mat_mul<<<grid, block>>>(g_mat1, g_mat2, g_mat_result, M, K, N);

    // 将结果从设备复制到主机
    cudaMemcpy(result, g_mat_result, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result matrix (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    // 释放内存
    free(mat1);
    free(mat2);
    free(result);
    cudaFree(g_mat1);
    cudaFree(g_mat2);
    cudaFree(g_mat_result);

    return 0;
}
