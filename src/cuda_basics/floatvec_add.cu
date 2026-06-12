#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA kernel for half-precision float vector addition
__global__ void floatVecAddHalf(const __half* A, const __half* B, __half* C, int N) {
    // 每个thread处理一个元素，支持Grid-Stride Loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < N; 
         idx += gridDim.x * blockDim.x) {
        // 将半精度转换为单精度，进行加法，再转回半精度
        float a = __half2float(A[idx]);
        float b = __half2float(B[idx]);
        C[idx] = __float2half(a + b);
    }
}

int main() {
    const int N = 1024 * 1024; // 向量大小: 1M elements
    size_t bytes = N * sizeof(__half);

    std::cout << "半精度浮点向量相加 (FP16)" << std::endl;
    std::cout << "向量大小: " << N << " elements" << std::endl;
    std::cout << "内存占用: " << (bytes / (1024 * 1024)) << " MB" << std::endl;

    // 分配主机内存
    __half* h_A = (__half*)malloc(bytes);
    __half* h_B = (__half*)malloc(bytes);
    __half* h_C = (__half*)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        std::cerr << "主机内存分配失败!" << std::endl;
        return 1;
    }

    // 初始化主机数据
    for (int i = 0; i < N; ++i) {
        h_A[i] = __float2half(1.5f);
        h_B[i] = __float2half(2.5f);
    }

    // 分配GPU内存
    __half *d_A, *d_B, *d_C;
    if (cudaMalloc((void**)&d_A, bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_B, bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_C, bytes) != cudaSuccess) {
        std::cerr << "GPU内存分配失败!" << std::endl;
        return 1;
    }

    // 将数据从主机复制到GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 启动kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    std::cout << "\n启动kernel: gridSize=" << gridSize << ", blockSize=" << blockSize << std::endl;
    
    floatVecAddHalf<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 检查kernel执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel执行失败: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // 同步GPU
    cudaDeviceSynchronize();

    // 将结果复制回主机
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    float expected = 4.0f; // 1.5 + 2.5 = 4.0
    for (int i = 0; i < N; ++i) {
        float result = __half2float(h_C[i]);
        // 半精度浮点精度约为千分位，所以容差设置为0.01
        if (fabs(result - expected) > 0.01f) {
            std::cerr << "验证失败: h_C[" << i << "] = " << result 
                      << ", 期望值 = " << expected << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "\n✓ 半精度浮点向量相加成功!" << std::endl;
        std::cout << "样本值: " << __half2float(h_C[0]) << std::endl;
    } else {
        std::cout << "\n✗ 半精度浮点向量相加失败!" << std::endl;
    }

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return success ? 0 : 1;
}