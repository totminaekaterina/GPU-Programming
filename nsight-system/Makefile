TARGET = 4_lab
# TARGET = 5

# Компилятор NVCC для CUDA
NVCC = nvcc

# Флаги компиляции
CFLAGS = `pkg-config --cflags opencv4`

# Флаги линковки (добавлено -lcublas)
LDFLAGS = `pkg-config --libs opencv4` -lcublas

# Флаги подавления предупреждений
DIAG_FLAGS = -diag-suppress=611,997

# Исходный файл
SRC = 4_lab.cu
# SRC = 5.cu


all: build


build:
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS) $(LDFLAGS) $(DIAG_FLAGS)


run: build
	./$(TARGET)

profile: build
	nsys profile --cudabacktrace=true --cuda-memory-usage=true -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o profile_report --force-overwrite=true ./$(TARGET)


clean:
	rm -f $(TARGET)
	rm -f *.qdrep *.sqlite


# export PATH=/usr/local/nsight-systems-2023.4.1/bin:$PATH