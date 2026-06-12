#include <cuda_runtime.h>
#include <iostream>

// simple add kernel (same as before)
__global__ void addKernel(const float *A, const float *B, float *C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

int main() {
  const int N = 1 << 20; // 1M elements
  size_t bytes = N * sizeof(float);

  // host allocation
  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);
  for (int i = 0; i < N; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // device allocation
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  // create two streams
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  cudaEvent_t evt;
  cudaEventCreate(&evt);
  // no inter-stream event needed for this independent example

  // begin capture on stream1 in global mode – commands on all streams will be
  // recorded
  cudaGraph_t graph;
  cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);

  // stream 1 work: copy A, add 1 to each element, copy back to h_C
  cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, s1);
  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  // we reuse d_C for stream1 output
  addKernel<<<blocks, threads, 0, s1>>>(d_A, d_A, d_C, N); // result = A + A

  cudaEventRecord(evt, s1); // 在捕获中，这会成为图的边

  cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, s1);

  // 流 B:
  cudaStreamWaitEvent(s2, evt,
                      0); // 在捕获中，会成为图的依赖：B 等待 A 完成
  // stream 2 work: copy B, add 2 to each element, copy back to h_B (reuse as
  // output)
  cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, s2);
  addKernel<<<blocks, threads, 0, s2>>>(d_B, d_B, d_C,
                                        N); // reuse d_C as temporary
  cudaMemcpyAsync(h_B, d_C, bytes, cudaMemcpyDeviceToHost, s2);

  // end capture (call on capturing stream)
  cudaStreamEndCapture(s1, &graph);

  // instantiate
  cudaGraphExec_t exec;
  cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

  // launch multiple times
  std::cout << "launching multi-stream graph 2 times" << std::endl;
  for (int i = 0; i < 2; ++i) {
    cudaGraphLaunch(exec, s1); // submit on any stream
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
  }

  // verify results: after parallel work, stream1 output in h_C should be 2.0,
  // stream2 output in h_B should be 4.0
  bool ok = true;
  for (int i = 0; i < N; ++i) {
    if (fabs(h_C[i] - 2.0f) > 1e-5 || fabs(h_B[i] - 4.0f) > 1e-5) {
      ok = false;
      if (i < 5) {
        std::cout << "mismatch at " << i << ": h_C=" << h_C[i]
                  << " h_B=" << h_B[i] << std::endl;
      }
    }
  }
  std::cout << (ok ? "verification passed" : "verification failed")
            << std::endl;

  // cleanup
  cudaGraphDestroy(graph);
  cudaGraphExecDestroy(exec);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return ok ? 0 : 1;
}