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
    int x = blockIdx.x * blockDim.x + threadIdx.x; // абсолютные координаты текущего потока в изображении
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
        output[(y * width + x) * channels + 3] = 255; // Альфа-канал
    }
}

// Измерение времени
template <typename KernelFunc, typename... Args>
void measureKernelExecutionTime(const char* message, KernelFunc kernel, dim3 gridSize, dim3 blockSize, Args... args) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<gridSize, blockSize>>>(args...); 
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << message << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " mcs" << std::endl;
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

    // Преобразование изображения в формат RGBA 
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    }
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    // size_t imageSize = width * height * channels * sizeof(unsigned char);

    // Определение количества доступных устройств
    int deviceCount;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));
    std::cout << "Number of available GPUs: " << deviceCount << std::endl;

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPUs found." << std::endl;
        return EXIT_FAILURE;
    }

    // Разделение данных по столбцам вместо строк
    int colsPerDevice = width / deviceCount;
    std::vector<unsigned char*> d_outputs(deviceCount); // вектор изображения
    std::vector<cudaArray*> cuArrays(deviceCount); // вектор cuda массивов 
    std::vector<cudaTextureObject_t> texObjs(deviceCount);
    std::vector<cudaStream_t> streams(deviceCount); // вектор из потоков на кажой из gpu

    for (int device = 0; device < deviceCount; ++device) {
        int startCol = device * colsPerDevice;
        int endCol = (device == deviceCount - 1) ? width : startCol + colsPerDevice;
        // если девайс последний, то он равен deviceCount - 1, колонка остановились == конец изображение
        // if (device == deviceCount - 1):
        //      endCol = width
        // else:
        //      endCol = startCol + colsPerDevice
        int colsForDevice = endCol - startCol; 
        // если изображение по ширине нечетное, 

        // Пропускаем GPU, если нет столбцов для обработки
        if (colsForDevice <= 0) {
            std::cout << "Skipping GPU " << device << " due to no columns to process" << std::endl;
            continue;
        }

        size_t deviceImageSize = colsForDevice * height * channels * sizeof(unsigned char); 
        // это просто размер в битах
        // площадь прямоугольника * количество цветов * сколько памяти нужно на один цвет (1 цвет хранит 1 байт)
        // получаем размер в байтах

        // Вывод параметров устройства
        std::cout << "GPU " << device << ": startCol = " << startCol 
                  << ", endCol = " << endCol 
                  << ", colsForDevice = " << colsForDevice << std::endl;


        // Установка текущего устройства
        cudaSetDevice(device);

        // Создание потока
        cudaStreamCreate(&streams[device]);
        // потоки - это разные процесс которые будет делать gpus 

        // Создание текстуры
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK_ERROR(cudaMallocArray(&cuArrays[device], &channelDesc, colsForDevice, height));
        // количество бит на 1 пиксель (&channelDesc)
        // ширина матрицы
        // высота матрицы

        // Копирование данных в массив
        CUDA_CHECK_ERROR(cudaMemcpy2DToArrayAsync(
            cuArrays[device], 0, 0, 
            image.data + startCol * channels, 
            width * channels * sizeof(unsigned char), 
            colsForDevice * channels * sizeof(unsigned char), 
            height, cudaMemcpyHostToDevice, streams[device]));
        // асинхронная функция, т.к. гпу могут работать асинхронно (1 гпу работает, потом 2)
        // начинаем копировани матрицы и пока оно копируется продолжаем функцию
        

        // Создание текстурного объекта
        cudaResourceDesc resDesc = {}; 
        // слева тип данных, справа - переменная пустой массив (длины 0)
        resDesc.resType = cudaResourceTypeArray;
        // заполнение полей resDesc, cudaResourceTypeArray - констранта
        resDesc.res.array.array = cuArrays[device];

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        // заполнили 5 полей
        // определеяем структуру класса


        CUDA_CHECK_ERROR(cudaCreateTextureObject(&texObjs[device], &resDesc, &texDesc, nullptr));
        // nullptr == None

        // Выделение памяти для выходных данных
        CUDA_CHECK_ERROR(cudaMalloc(&d_outputs[device], deviceImageSize));
        // выделяет одномерный массив в памяти gpu

        // Настройка размеров сетки
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // тк 2d изображение
        dim3 gridSize((colsForDevice + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        // округление вверх (5 + 3-1) / 3, т.к. если вниз часть изображения никуда не попадет

        // Вывод размеров сетки и блоков
        std::cout << "GPU " << device << ": gridSize = (" 
                << gridSize.x << ", " << gridSize.y << "), blockSize = (" 
                << blockSize.x << ", " << blockSize.y << ")" << std::endl;

        // Запуск ядра
        measureKernelExecutionTime(
            ("GPU " + std::to_string(device) + " Execution Time").c_str(),
            sobelFilterTexture,
            gridSize, blockSize, texObjs[device], d_outputs[device], colsForDevice, height, channels);

        // Синхронизация потока
        CUDA_CHECK_ERROR(cudaStreamSynchronize(streams[device]));
    
    }

    // Сохранение результата
    const std::string outputFilename = "./output_textured_sobel_multi.png";
    saveImage(outputFilename, image);

    return 0;
}




