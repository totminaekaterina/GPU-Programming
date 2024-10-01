// cuda_runtime.ipynb

// !nvcc --version

// !apt-get install -y nvidia-cuda-toolkit

%%writefile example.cu
include <iostream>
include <cmath>
include <vector>
include <cuda_runtime.h>

define M_PI 3.14159265358979323846


// CPU для float
std::vector<float> float_sin_cpu(int N) {
    std::vector<float> result(N);
    for (int i = 0; i < N; ++i) {
        float angle = (i % 360) * M_PI / 180.0f;
        result[i] = sinf(angle); // возвращает float
    }
    return result;
}


// CPU для double
std::vector<float> double_sin_cpu(int N) {
    std::vector<float> result(N);
    for (int i = 0; i < N; ++i) {
        float angle = (i % 360) * M_PI / 180.0f;
        result[i] = sin(angle); // sin возвращает double
    }
    return result; // возвращает вектор
}



// CUDA ядро для float
__global__ void compute_sin_float(float* d_result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float angle = (i % 360) * M_PI / 180.0f;
        d_result[i] = __sinf(angle); // __sinf возвращает float
    }
}


// CUDA ядро для double
__global__ void compute_sin_double(double* d_result, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double angle = (i % 360) * M_PI / 180.0;
        d_result[i] = sin(angle);
    }
}


// Вычисление ошибки
template<class Ty1, class Ty2>
double calculate_error(const std::vector<Ty1>& arr1, const std::vector<Ty2>& arr2)
{
	double error = 0.0;
	size_t N = arr1.size();
	for (size_t i = 0; i < N; ++i)
	{
		error += std::abs(static_cast<double>(arr1[i]) - static_cast<double>(arr2[i]));
	}
	return error / N;
}




int main() {
    const int N = 360; // 1e9

    // CPU
    // float
    auto float_result_cpu = float_sin_cpu(N);
    // double
    auto double_result_cpu = double_sin_cpu(N);




    // GPU float
    float* d_float_result; //  хранение результатов на GPU после вычисления синуса для float
    float* h_float_result = (float*)malloc(N * sizeof(float)); // выделение памяти на CPU хосте

    cudaMalloc(&d_float_result, N * sizeof(float));

    int threadIdx_num = 256;
    int blockDim_num = (N + threadIdx_num - 1) / threadIdx_num;


    // Запуск compute_sin_float
    compute_sin_float<<<blockDim_num, threadIdx_num>>>(d_float_result, N);


    // cudaMemcpyDeviceToHost указывает, что данные копируются с GPU на CPU хост

    cudaMemcpy(h_float_result, d_float_result, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_float_result);



    // GPU double
    double* d_double_result; //  хранение результатов на GPU полсе вычисления синуса для double
    double* h_double_result = (double*)malloc(N * sizeof(double));
    cudaMalloc(&d_double_result, N * sizeof(double));

    compute_sin_double<<<blockDim_num, threadIdx_num>>>(d_double_result, N);
    cudaMemcpy(h_double_result, d_double_result, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_double_result);


    // float - float / double - double
    double float_error = calculate_error(float_result_cpu, std::vector<float>(h_float_result, h_float_result + N));
    double double_error = calculate_error(double_result_cpu, std::vector<double>(h_double_result, h_double_result + N));

    // float - double / double - float
    double float_double_error = calculate_error(float_result_cpu, std::vector<double>(h_double_result, h_double_result + N));
    double double_float_error = calculate_error(double_result_cpu, std::vector<float>(h_float_result, h_float_result + N));


    // общая для всех синусов
    // double total_error = (float_error + double_error + float_double_error + double_float_error) / 4.0;

    // std::cout << "Double Error (CPU float vs GPU float): " << float_error << std::endl;
     std::cout << "Double Error (CPU double vs GPU double): " << double_error << std::endl;
    // std::cout << "Double Error (CPU float vs GPU double): " << float_double_error << std::endl;
    // std::cout << "Double Error (CPU double vs GPU float): " << double_float_error << std::endl;


    // std::cout << "Total Error (Average of all errors): " << total_error << std::endl;

    free(h_float_result);
    free(h_double_result);

    return 0;
}

// !nvcc -o example example.cu

// !./example



