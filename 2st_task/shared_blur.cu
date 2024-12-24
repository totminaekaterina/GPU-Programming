#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <unistd.h>

#define CUDA_DEBUG

#ifdef CUDA_DEBUG
#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
    printf("Cuda error: %s\n", cudaGetErrorString(err));    \
    printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}
#else
#define CUDA_CHECK_ERROR(err)
#endif



#define BLOCK_SIZE 16
#define FILTER_SIZE 3
#define TILE_SIZE (BLOCK_SIZE - FILTER_SIZE + 1)


__global__ void blurFilterShared(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    // Определение размера разделяемой памяти с учетом фильтра 3x3
    extern __shared__ unsigned char sharedMem[];
    int sharedWidth = blockDim.x + FILTER_SIZE - 1;
    int sharedHeight = blockDim.y + FILTER_SIZE - 1;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx; 
    int y = blockIdx.y * TILE_SIZE + ty;

    int sharedX = tx + FILTER_SIZE / 2; 
    int sharedY = ty + FILTER_SIZE / 2;

    // Загружаем данные в shared memory с учетом границ
    for (int c = 0; c < channels; ++c) { 
        if (x < width && y < height) {
            sharedMem[(sharedY * sharedWidth + sharedX) * channels + c] = input[(y * width + x) * channels + c];
        } else {
            sharedMem[(sharedY * sharedWidth + sharedX) * channels + c] = 0;
        }

        // Левая граница
        if (tx < FILTER_SIZE / 2 && x >= FILTER_SIZE / 2) {
            sharedMem[(sharedY * sharedWidth + tx) * channels + c] = input[(y * width + (x - FILTER_SIZE / 2)) * channels + c];
        } else if (tx < FILTER_SIZE / 2) {
            sharedMem[(sharedY * sharedWidth + tx) * channels + c] = 0;
        }

        // Верхняя граница
        if (ty < FILTER_SIZE / 2 && y >= FILTER_SIZE / 2) {
            sharedMem[(ty * sharedWidth + sharedX) * channels + c] = input[((y - FILTER_SIZE / 2) * width + x) * channels + c];
        } else if (ty < FILTER_SIZE / 2) {
            sharedMem[(ty * sharedWidth + sharedX) * channels + c] = 0;
        }

        // Правая граница
        if (tx >= blockDim.x - FILTER_SIZE / 2 && (x + FILTER_SIZE / 2) < width) {
            sharedMem[(sharedY * sharedWidth + sharedX + FILTER_SIZE / 2) * channels + c] = input[(y * width + (x + FILTER_SIZE / 2)) * channels + c];
        } else if (tx >= blockDim.x - FILTER_SIZE / 2) {
            sharedMem[(sharedY * sharedWidth + sharedX + FILTER_SIZE / 2) * channels + c] = 0;
        }

        // Нижняя граница
        if (ty >= blockDim.y - FILTER_SIZE / 2 && (y + FILTER_SIZE / 2) < height) {
            sharedMem[((sharedY + FILTER_SIZE / 2) * sharedWidth + sharedX) * channels + c] = input[((y + FILTER_SIZE / 2) * width + x) * channels + c];
        } else if (ty >= blockDim.y - FILTER_SIZE / 2) {
            sharedMem[((sharedY + FILTER_SIZE / 2) * sharedWidth + sharedX) * channels + c] = 0;
        }
    }

    __syncthreads();


    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            int sum = 0;
            int count = 0;
            for (int r = -FILTER_SIZE / 2; r <= FILTER_SIZE / 2; ++r) {
                for (int d = -FILTER_SIZE / 2; d <= FILTER_SIZE / 2; ++d) {
                    int shared_x = sharedX + d;
                    int shared_y = sharedY + r;

                    if (shared_x >= 0 && shared_x < sharedWidth &&
                        shared_y >= 0 && shared_y < sharedHeight) {

                        sum += sharedMem[(shared_y * sharedWidth + shared_x) * channels + c];
                        count++;
                    }
                }
            }

            // Запись результата обратно в глобальную память
            output[(y * width + x) * channels + c] = sum / count;
        }
        
    }
}


// Измерение времени
template <typename KernelFunc, typename... Args>
void measureKernelExecutionTime(const char* message, KernelFunc kernel, dim3 gridSize, dim3 blockSize, size_t sharedMemSize, Args... args) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<gridSize, blockSize, sharedMemSize>>>(args...); 
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << message << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}


// Загрузка изображения
void loadImage(const std::string& filename, cv::Mat& image) {
    image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Image is empty " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Image is loaded. Size: " << image.cols << "x" << image.rows << ", Channels num: " << image.channels() << std::endl;
}

// Сохранение изображения
void saveImage(const std::string& filename, const cv::Mat& image) {
    if (!cv::imwrite(filename, image)) {
        std::cerr << "The error during saving image " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Output image was saved successfully " << filename << std::endl;
}

// Выделение памяти и копирование данных на GPU
void allocateAndCopyToDevice(unsigned char* h_data, unsigned char** d_data, size_t size) {
    CUDA_CHECK_ERROR(cudaMalloc(d_data, size));
    CUDA_CHECK_ERROR(cudaMemcpy(*d_data, h_data, size, cudaMemcpyHostToDevice));
}

int main() {
    const std::string filename = "/cuda/image.png";
    cv::Mat image;
    loadImage(filename, image);

    // Преобразование изображения в формат RGBA
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    size_t imageSize = width * height * channels * sizeof(unsigned char);

    // Объявление указателей для данных на GPU
    unsigned char* d_input;
    unsigned char* d_output;

    // Выделение и копирование данных на GPU
    allocateAndCopyToDevice(image.data, &d_input, imageSize);
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, imageSize));

    // Настройка размера блока и сетки
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    size_t sharedMemSize = (BLOCK_SIZE + FILTER_SIZE - 1) * (BLOCK_SIZE + FILTER_SIZE - 1) * channels * sizeof(unsigned char);

    measureKernelExecutionTime("Time spend for Shared Memory (Sobel filter) ", blurFilterShared, gridSize, blockSize, sharedMemSize, d_input, d_output, width, height, channels);
    
    // Копирование результата обратно на хост
    CUDA_CHECK_ERROR(cudaMemcpy(image.data, d_output, imageSize, cudaMemcpyDeviceToHost));

    saveImage("./output_shared_blur.png", image);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}