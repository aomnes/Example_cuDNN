#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
            << image.channels() << std::endl;
  return image;
}

void save_image(const char* output_filename,
                float* buffer,
                int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  // Make negative values zero.
  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
  std::cerr << "Wrote output to " << output_filename << std::endl;
}

int main(int argc, const char* argv[]) {
  std::cout << "CUDNN_VERSION: " << CUDNN_VERSION << std::endl;

  if (argc < 2) {
    std::cerr << "usage: conv <image> [gpu=0]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;

  cv::Mat image = load_image(argv[1]);

  cudaSetDevice(gpu_id);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));


  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NHWC,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/1,
                                        /*channels=*/3,
                                        /*image_height=*/image.rows,
                                        /*image_width=*/image.cols));


    cudnnPoolingDescriptor_t pooling_descriptor;
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    checkCUDNN(cudnnSetPooling2dDescriptor(
        pooling_descriptor,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        /*windowHeight*/2,
        /*windowWidth*/2,
        /*verticalPadding*/0,
        /*horizontalPadding*/0,
        /*verticalStride*/1,
        /*horizontalStride*/1));

    int batch_size{1}, channels{3}, height{350}, width{292};

  int image_bytes = batch_size * channels * height * width * sizeof(float);
  int image_output_bytes = batch_size * channels * height/2 * width/2 * sizeof(float);

  float* d_input{nullptr};
  cudaMalloc(&d_input, image_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

  float* d_output{nullptr};
  cudaMalloc(&d_output, image_output_bytes);
  cudaMemset(d_output, 0, image_output_bytes);

  const float alpha = 1.0f, beta = 0.0f;

  checkCUDNN(cudnnPoolingForward(cudnn,
      pooling_descriptor,
      &alpha,
      input_descriptor,
      d_input,
      &beta,
      output_descriptor,
      d_output));

  float* h_output = new float[image_output_bytes];
  cudaMemcpy(h_output, d_output, image_output_bytes, cudaMemcpyDeviceToHost);

  save_image("cudnn-out.png", h_output, height, width);

  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyPoolingDescriptor(pooling_descriptor);

  cudnnDestroy(cudnn);
}
