#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

#define BASE_TYPE double

void matrixMultiplyCUBLAS(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, int Arows, int Acols, int Bcols) {
    cublasHandle_t handle;
    cublasCreate(&handle); // инициализаия кублас, создает контекст

    const BASE_TYPE alpha = 1.0;
    const BASE_TYPE beta = 0.0; 

    //  С = alpha * A (Arows x Acols) * B (Acols x Bcols) + beta * С
    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        Arows, Bcols, Acols, // Arows==Crows, Bcols==Ccols, Acols==Brows
        &alpha,
        A, Arows,   // A 2x3, массив одномерный (сколько элементов нужно отсутпить чтобы перейти на следующую колонку для матрицы A)
        B, Acols,   // B 3x4, массив одномерный (сколько элементов нужно отсутпить чтобы перейти на следующую колонку для матрицы B)
        &beta,
        C, Arows);  // C - 2x4

    cublasDestroy(handle);
}

// создаем матрицу, конструктор
class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Rows and columns must be positive.");
        }
        data = new double[rows * cols]; // Allocate memory for the matrix
        // new == malloc, new автоматом дает sizeof
    }

    ~Matrix() {
        delete[] data; // Free allocated memory
    }

    void set(int r, int c, double value) {
        if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
            throw std::out_of_range("Index out of range.");
        }
        data[r + c * rows_] = value; // Column-major order
    }

    // Get the value at (i, j)
    double get(int r, int c) const {
        if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
            throw std::out_of_range("Index out of range.");
        }
        return data[r + c * rows_]; // Column-major order
    }

    // Print the matrix for debugging
    void print() const {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                std::cout << get(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    int rows_;      // Number of rows
    int cols_;      // Number of columns
    double* data; 
};

int main() {
    int Arows = 1000;
    int Acols = 2000;
    Matrix A(Arows, Acols);

    // Set values in the matrix
    for (int r = 0; r < Arows; ++r) {
        for (int c = 0; c < Acols; ++c) {
            // A.set(r, c, 1); // Example values
            A.set(r, c, rand() / (BASE_TYPE)RAND_MAX);
        }
    }
    int Brows = Acols;
    int Bcols = 1500;
    Matrix B(Brows, Bcols);

    // Set values in the matrix
    for (int r = 0; r < Brows; ++r) {
        for (int c = 0; c < Bcols; ++c) {
            // B.set(r, c, c+r); // Example values
            B.set(r, c, rand() / (BASE_TYPE)RAND_MAX);
        }
    }

    int Crows = Arows;
    int Ccols = Bcols;
    Matrix C(Crows, Ccols);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Crows * Ccols * sizeof(BASE_TYPE);

    BASE_TYPE* h_A = A.data;
    BASE_TYPE* h_B = B.data;
    BASE_TYPE* h_C = C.data;


    // A.print();
    // B.print();

    BASE_TYPE* d_A = NULL;
    cudaMalloc((void**)&d_A, Asize);

    BASE_TYPE* d_B = NULL;
    cudaMalloc((void**)&d_B, Bsize);

    BASE_TYPE* d_C = NULL;
    cudaMalloc((void**)&d_C, Csize);

    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    matrixMultiplyCUBLAS(d_A, d_B, d_C, Arows, Acols, Bcols);

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    
    cudaEventElapsedTime(&milliseconds, start, stop);


    std::cout << "Matrix multiplication took " << milliseconds << " mc." << std::endl;

    cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);

    // C.print();

    std::cout << "Test STARTED" << std::endl;

    // 1 5
    // 2 3
    // 
    // 2 3 1 
    // 8 9 4
    // 
    // 1*2 + 5*8 1*3 + 5*9 1*1 + 5*4
    // 2*2 + 3*8 2*3 + 3*9 2*1 + 3*4

    for (int i = 0; i < Crows; i++) {
        for (int j = 0; j < Ccols; j++) {
            BASE_TYPE sum = 0;
            for (int k = 0; k < Acols; k++)
                sum += A.get(i, k) * B.get(k, j);
             if (fabs(C.get(i, j) - sum) > 1e-3) {
                 fprintf(stderr, "Result verification failed at element [%d, %d]!\n", i, j);
                 printf("sum = %lf, C.get(i, j) = %lf\n", sum, C.get(i, j));
                 exit(EXIT_FAILURE);
             }
            // printf("%lf ", sum);
        }
        // printf("\n ");
    }

    std::cout << "Test PASSED" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
