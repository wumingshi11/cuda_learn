#include <iostream>
#include <cuda_runtime.h>

// simple vector add kernel
__global__ void addKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1<<20; // 1M elements
    size_t bytes = N * sizeof(float);

    // host allocation
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // device allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // create a stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // begin capture: subsequent work into a cudaGraph
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // operations recorded in the stream
    cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    addKernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, N);

    cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, stream);

    // end capture and create executable graph
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // now we can launch the graph as many times as needed with low overhead
    std::cout << "Launching graph 3 times" << std::endl;
    for (int i = 0; i < 3; ++i) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }

    // cleanup
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    std::cout << "Done" << std::endl;
    return 0;
}