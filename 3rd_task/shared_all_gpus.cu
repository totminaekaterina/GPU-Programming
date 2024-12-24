#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <unistd.h>
#include <thread>

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
    int sharedWidth = blockDim.x + FILTER_SIZE - 1; //blockDim.x - размер блока по ширине
    int sharedHeight = blockDim.y + FILTER_SIZE - 1; //blockDim.y - размер блока по высоте

    int tx = threadIdx.x; // индекс текущего потока в блоке по оси x
    int ty = threadIdx.y; // индекс текущего потока в блоке по оси y
    int x = blockIdx.x * TILE_SIZE + tx; // координата по x текущего потока, который обрабатывает какой то пиксель на изображении
    int y = blockIdx.y * TILE_SIZE + ty; //

    int sharedX = tx + FILTER_SIZE / 2; // позиция потока по x в разделяемой памяти
    int sharedY = ty + FILTER_SIZE / 2;

// (0,0)R, (0,0)G, (0,0)B, (1,0)R...

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


template <typename KernelFunc, typename... Args>
void measureKernelExecutionTime(const char* message, KernelFunc kernel, dim3 gridSize, dim3 blockSize, size_t sharedMemSize, Args... args) {
    // Запуск таймера
    auto start = std::chrono::high_resolution_clock::now();
    // Запуск ядра
    kernel<<<gridSize, blockSize, sharedMemSize>>>(args...);
    // Синхронизация устройства
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    // Остановка таймера
    auto end = std::chrono::high_resolution_clock::now();
    // Вывод времени выполнения в микросекундах
    std::cout << message << ": " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " mcs" << std::endl;
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

// (0,0)R, (0,0)G, (0,0)B, (1,0)R, (1,0)G, (1,0)B, ...

// Выделение памяти и копирование данных на GPU
void allocateAndCopyToDevice(unsigned char* h_data, unsigned char** d_data, size_t size) {
    CUDA_CHECK_ERROR(cudaMalloc(d_data, size));
    CUDA_CHECK_ERROR(cudaMemcpy(*d_data, h_data, size, cudaMemcpyHostToDevice));
    // куда откуда сколько, cudaMemcpyHostToDevice - константа откуда куда копировать
}
//  unsigned char* - тип данных
// h_data - память на хосте, название переменной
// unsigned char** - двумерный массив, d_data - название переменной
// size_t size - size_t - беззнаковый int (на 32, 64 - разное, int такой какая размерность системы, зависит от компилятора)

// Основная функция обработки на GPU
// i, numGPUs, h_inputImage, h_outputImage, width, height, channels
void processOnGPU(int deviceID, int numGPUs, unsigned char* h_inputImage, unsigned char* h_outputImage, 
                  int width, int height, int channels) {
    cudaSetDevice(deviceID); // настраиваем программу cuda используя конкретный девайс

    int overlap = FILTER_SIZE / 2; // Перекрытие для краев на 2 (слева - справа)
    int segmentHeight = height / numGPUs;
    int yOffset = segmentHeight * deviceID;
    
    // релевантно только для id = 1
    if (deviceID == numGPUs - 1) {
        segmentHeight += height % numGPUs;
    }

    int startRow = (deviceID == 0) ? 0 : yOffset - overlap;
    // if (deviceID == 0):
    //  startRow = 0
    // else:
    //  startRow = yOffset - overlap

    int endRow = (deviceID == numGPUs - 1) ? height : yOffset + segmentHeight + overlap;
    // if (deviceID == numGPUs - 1):
    //  endRow = height
    // else:
    //  endRow = yOffset + segmentHeight + overlap
    int segmentHeightWithOverlap = endRow - startRow;

    size_t segmentSizeWithOverlap = width * segmentHeightWithOverlap * channels * sizeof(unsigned char);
    unsigned char *d_inputImage, *d_outputImage; // объявила указатели

    cudaMalloc(&d_inputImage, segmentSizeWithOverlap);
    cudaMalloc(&d_outputImage, segmentSizeWithOverlap);
    // два массива под 2 изображения - инпут и аутпут

    cudaMemcpy(d_inputImage, &h_inputImage[startRow * width * channels], segmentSizeWithOverlap, cudaMemcpyHostToDevice);
    //  если startRow = 0 - беру все изображение, если startRow !=0 то пропускаем первый startRow в зависимости от условий выше
    // &h_inputImage[startRow * width * channels] - откуда
    // segmentSizeWithOverlap - количество

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (segmentHeightWithOverlap + TILE_SIZE - 1) / TILE_SIZE);
    size_t sharedMemSize = (BLOCK_SIZE + FILTER_SIZE - 1) * (BLOCK_SIZE + FILTER_SIZE - 1) * channels * sizeof(unsigned char);
    // const unsigned char* input, unsigned char* output, int width, int height, int channels
    // Измерение времени выполнения ядра

    std::string gpuMessage = "GPU " + std::to_string(deviceID) + " execution time";
    measureKernelExecutionTime(gpuMessage.c_str(), blurFilterShared, gridSize, blockSize, sharedMemSize, 
                               d_inputImage, d_outputImage, width, segmentHeightWithOverlap, channels);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Копирование центральной части (без перекрытия) обратно на хост
    int copyStartRow = (deviceID == 0) ? 0 : overlap;
    // if (deviceID == 0):
    //      copyStartRow = 0
    // else:
    //      copyStartRow = overlap
    
    int copyEndRow = (deviceID == numGPUs - 1) ? segmentHeightWithOverlap : segmentHeightWithOverlap - overlap;
    // if (deviceID == numGPUs - 1):
    //      copyEndRow = segmentHeightWithOverlap
    // else:
    //      copyEndRow = segmentHeightWithOverlap - overlap

    int copyHeight = copyEndRow - copyStartRow;
    // первую и последнюю строки сколько реальн строк осталось
    size_t centralSegmentSize = width * copyHeight * channels * sizeof(unsigned char);
    // количество байт в ядре без учета overlap

    cudaMemcpy(&h_outputImage[yOffset * width * channels], &d_outputImage[copyStartRow * width * channels], centralSegmentSize, cudaMemcpyDeviceToHost);
    // &h_outputImage[yOffset * width * channels] - указатель на конкретный пиксель изображения, которое НАЧАЛО строки yOffset (хост)
    // &d_outputImage[copyStartRow * width * channels] - указатель на конкретный пиксель изображения, которое НАЧАЛО строки copyStartRow (девайс)
    // количество centralSegmentSize в байтах
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}


// Основная функция обработки
void processOnMultipleGPUs(const cv::Mat& image, cv::Mat& outputImage) {
    int numGPUs;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&numGPUs));
    if (numGPUs == 0) {
        std::cerr << "No CUDA-compatible devices found" << std::endl;
        return;
    }

    std::cout << "Number of GPUs available: " << numGPUs << std::endl;

    int width = image.cols; // поле считываю
    int height = image.rows; // поле считываю
    int channels = image.channels(); // функция, возвращает количество каналов
    // size_t imageSize = width * height * channels * sizeof(unsigned char);

    unsigned char* h_inputImage = image.data; // создаю переменную с типом указателем, приравниваю значение 
    // (0,0)R, (0,0)G, (0,0)B, (1,0)R, (1,0)G, (1,0)B, ...
    
    unsigned char* h_outputImage = outputImage.data;
    // (0,0)R, (0,0)G, (0,0)B, (1,0)R, (1,0)G, (1,0)B, ...
    
    std::vector<std::thread> threads;
    // вектор - это массив переменных, тип которых - поток std::thread
    // std::vector<int> - вектор интов

    // Запуск обработки на каждом GPU в отдельных потоках
    for (int i = 0; i < numGPUs; i++) {
        threads.emplace_back(processOnGPU, i, numGPUs, h_inputImage, h_outputImage, width, height, channels);
    }

    // threads[0] - первый поток
    // i - deviceid
    // нужно передать поток, берем processOnGPU(i, numGPUs, h_inputImage, h_outputImage, width, height, channels) - создаем паралельные потоки
    // Ожидание завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    const std::string filename = "/cuda/image.png";
    cv::Mat image;
    loadImage(filename, image);

    // if (image.channels() == 3) {
    //     cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    // }

    cv::Mat outputImage = cv::Mat::zeros(image.size(), image.type());

    processOnMultipleGPUs(image, outputImage);

    saveImage("/cuda/output_shared_blur_multi_gpu.png", outputImage);

    return 0;
}



