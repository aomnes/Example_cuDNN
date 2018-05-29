CXX := nvcc
TARGET := pooling
CUDNN_PATH := cudnn
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_61 -ccbin gcc-4.9 -std=c++11 -O2

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lstdc++

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
