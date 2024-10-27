# Description
## Описание задачи

Выделить на GPU массив arr из 10^9 элементов типа float и инициализировать его с помощью ядра следующим образом: arr[i] = sin((i%360)*Pi/180). Скопировать массив в память центрального процессора и посчитать ошибку err = sum_i(abs(sin((i%360)*Pi/180) - arr[i]))/10^9. Провести исследование зависимости результата от использования функций: sin, sinf, __sinf. Объяснить результат. Проверить результат при использовании массива типа double.

## Полученные результаты
| Methods (1st ↔ 2nd)     | Time for every Methos) (1st vs 2nd) | Double Error |
| ----------------------- | ----------------------------------- | ------------ |
| CPU float ↔ GPU float   | 2.9e-05 sec vs 0.094204 sec         | 1.22381e-07  |
| CPU double ↔ GPU double | 1.1e-05 sec vs 0.000308 sec         | 4.56898e-08  |
| GPU float ↔ GPU double  | 1.9e-05 sec vs 0.000192 sec         | 4.56898e-08  |
| GPU double ↔ GPU float  | 1.1e-05 sec vs 0.113997 sec         | 1.22381e-07  |


Как результат, вычисления на GPU проходят намного быстрее, чем на CPU, но точность вычислений на GPU сравнительно ниже, чем на CPU.

## Функции
`std::vector<float> float_sin_cpu(int N)`
- Функция вычисляет значения синуса для углов от 0 до N-1 градусов, используя стандартную библиотеку C++ и функцию sinf для работы с числами типа float.
- Возвращаемое значение: Вектор, содержащий значения синуса для каждого угла.

`__global__ void compute_sin_float(float* d_result, int N)`

- CUDA-ядро, которое выполняет вычисления синуса для углов от 0 до N-1 градусов на GPU. Каждый поток обрабатывает одно значение.
Параметры:
- float* d_result: Указатель на массив в памяти GPU, где будут сохранены результаты.
- int N: Количество вычисляемых значений.

`template<class Ty1, class Ty2> double calculate_error(const std::vector<Ty1>& arr1, const std::vector<Ty2>& arr2)`
- Функция для вычисления средней абсолютной ошибки между двумя векторами значений. Используется для сравнения результатов вычислений на CPU и GPU.
- Возвращаемое значение: Средняя абсолютная ошибка.

### Использование памяти на GPU
- Выделение памяти: Используется функция cudaMalloc для выделения памяти на GPU, и malloc для выделения памяти на CPU.
- Копирование данных: Результаты вычислений синуса копируются с GPU на CPU с помощью cudaMemcpy.
- Освобождение памяти: Память на GPU освобождается после использования с помощью cudaFree.

### Параллелизм в CUDA
- Сетка блоков и потоков: Вычисления организованы в сетку блоков и потоков. Каждый поток вычисляет одно значение синуса.
- Индексация потоков: Уникальный индекс для каждого потока вычисляется с помощью blockIdx.x, blockDim.x и threadIdx.x.

### Пример использования
- CPU: Вычисления синуса выполняются последовательно с использованием float_sin_cpu.
- GPU: Вычисления синуса выполняются параллельно с использованием compute_sin_float.