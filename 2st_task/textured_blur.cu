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

// Размытие с использованием textured memory
__global__ void sobelFilterTexture(cudaTextureObject_t texObj, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        int sumX[3] = {0, 0, 0};
        int sumY[3] = {0, 0, 0};

        for (int r = -FILTER_SIZE / 2; r <= FILTER_SIZE / 2; ++r) {
            for (int d = -FILTER_SIZE / 2; d <= FILTER_SIZE / 2; ++d) {
                int sampledX = x + d;
                int sampledY = y + r;

                if (sampledX >= 0 && sampledX < width && sampledY >= 0 && sampledY < height) {
                    uchar4 sampledValue = tex2D<uchar4>(texObj, sampledX, sampledY);
                    sumX[0] += Gx[r + 1][d + 1] * sampledValue.x;
                    sumX[1] += Gx[r + 1][d + 1] * sampledValue.y;
                    sumX[2] += Gx[r + 1][d + 1] * sampledValue.z;

                    sumY[0] += Gy[r + 1][d + 1] * sampledValue.x;
                    sumY[1] += Gy[r + 1][d + 1] * sampledValue.y;
                    sumY[2] += Gy[r + 1][d + 1] * sampledValue.z;
                }
            }
        }

        // Вычисление результирующего значения градиента
        for (int c = 0; c < channels - 1; ++c) {
            int magnitude = min(255, (int)sqrtf((float)(sumX[c] * sumX[c] + sumY[c] * sumY[c])));
            output[(y * width + x) * channels + c] = magnitude;
        }
        output[(y * width + x) * channels + 3] = 255; // Альфа-канал (если имеется)
    }
}

// Измерение времени
template <typename KernelFunc, typename... Args>
void measureKernelExecutionTime(const char* message, KernelFunc kernel, dim3 gridSize, dim3 blockSize, Args... args) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<gridSize, blockSize>>>(args...); 
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << message << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

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

int main() {
    // Загрузка изображения
    const std::string filename = "/cuda/image.png";
    cv::Mat image;
    loadImage(filename, image);

    // Преобразование изображения в формат RGBA, если это необходимо
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    }
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    size_t imageSize = width * height * channels * sizeof(unsigned char);

    // Создание текстуры для входных данных
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* cuArray;
    CUDA_CHECK_ERROR(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // Копирование данных в массив с использованием cudaMemcpy2DToArray
    CUDA_CHECK_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, image.data, width * channels * sizeof(unsigned char), width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice));

    // Создание текстурного объекта
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // Выделение памяти для выходных данных
    unsigned char* d_output;
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, imageSize));

    // Настройка размера блока и сетки
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Запуск CUDA-ядра
    measureKernelExecutionTime("Time spend for Textured Memory (Sobel filter)", sobelFilterTexture, gridSize, blockSize, texObj, d_output, width, height, channels);

    // Копирование результата обратно на хост
    cv::Mat outputImage(height, width, CV_8UC4);
    CUDA_CHECK_ERROR(cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost));

    saveImage("./output_textured_sobel.png", outputImage);

    // Освобождение ресурсов
    CUDA_CHECK_ERROR(cudaDestroyTextureObject(texObj));
    CUDA_CHECK_ERROR(cudaFreeArray(cuArray));
    CUDA_CHECK_ERROR(cudaFree(d_output));

    return 0;
}
