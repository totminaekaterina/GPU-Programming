#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>


#define BLOCK_SIZE 16
#define BASE_TYPE double


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Ядро для перемножения матриц
__global__ void matrixMultKernel(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int Acols, int Bcols) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; // абсолютные координаты x текущего потока в матрице
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < Acols && col < Bcols) {
        BASE_TYPE sum = 0.0;
        for (int k = 0; k < Acols; ++k) {
            sum += A[row * Acols + k] * B[k * Bcols + col];
        }
        C[row * Bcols + col] = sum;
    }
}


// Функция выравнивания размера относительно блока
inline int alignToBlockSize(int size, int blockSize) {
    return (size + blockSize - 1) / blockSize * blockSize;
}


// Функция для инициализации матрицы случайными числами
void initializeMatrix(BASE_TYPE *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() / (BASE_TYPE)RAND_MAX; // генерация рандомного числа
    }
}


// Функция для проверки результатов
void validateResults(const BASE_TYPE *A, const BASE_TYPE *B, const BASE_TYPE *C, int Arows, int Acols, int Bcols) {
    for (int i = 0; i < Arows; ++i) {
        for (int j = 0; j < Bcols; ++j) {
            BASE_TYPE sum = 0.0;
            // проходим по столбцам и строкам результатирующей С
            // k - вычисление скалярного произведения i на j
            for (int k = 0; k < Acols; ++k) {
                sum += A[i * Acols + k] * B[k * Bcols + j];
            }
            // абсолютное занчение с запятой
            // задание порога
            if (fabs(C[i * Bcols + j] - sum) > 1e-3) {
                fprintf(stderr, "Mismatch at [%d, %d]: GPU=%.6f, CPU=%.6f\n", i, j, C[i * Bcols + j], sum);
                // 
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("Validation PASSED\n");
}


int main() {
    printf("Initializing matrices...\n");


    // Размеры матриц
    int Arows = 1000, Acols = 2000, Bcols = 1500;


    // Выравнивание размеров
    Arows = alignToBlockSize(Arows, BLOCK_SIZE);
    Acols = alignToBlockSize(Acols, BLOCK_SIZE);
    Bcols = alignToBlockSize(Bcols, BLOCK_SIZE);


    printf("Matrix dimensions (after alignment):\n");
    printf("  A: %d x %d\n", Arows, Acols);
    printf("  B: %d x %d\n", Acols, Bcols);


    // Выделение памяти на хосте
    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Acols * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);


    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE *)malloc(Csize);


    // Инициализация матриц
    initializeMatrix(h_A, Arows, Acols);
    initializeMatrix(h_B, Acols, Bcols);


    // Выделение памяти на устройстве
    BASE_TYPE *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, Asize));
    CUDA_CHECK(cudaMalloc((void **)&d_B, Bsize));
    CUDA_CHECK(cudaMalloc((void **)&d_C, Csize));


    // Копирование данных на устройство
    CUDA_CHECK(cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice));


    // Настройка параметров запуска ядра
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((Bcols + BLOCK_SIZE - 1) / BLOCK_SIZE, (Arows + BLOCK_SIZE - 1) / BLOCK_SIZE);


    // Запуск ядра и измерение времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));


    matrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Acols, Bcols);


    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));


    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Kernel execution time: %.2f ms\n", elapsedTime);


    // Копирование результата на хост
    CUDA_CHECK(cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost));


    // Проверка результатов
    validateResults(h_A, h_B, h_C, Arows, Acols, Bcols);


    // Освобождение ресурсов
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    printf("Program finished successfully.\n");
    return 0;
}





