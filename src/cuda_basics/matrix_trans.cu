#include <iostream>
#include <cuda_runtime.h>

// Tile width for shared memory
#define TILE_DIM 32
#define BLOCK_ROWS 8

// CUDA kernel using shared memory to perform matrix transpose
__global__ void transposeShared(const float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Load data into shared memory tile
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((y + i) < height && x < width) {
            tile[threadIdx.y + i][threadIdx.x] = input[(y + i) * width + x];
        }
    }

    __syncthreads();

    // Write transposed data to output
    x = blockIdx.y * TILE_DIM + threadIdx.x; // note swap of blockIdx.x/y
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((y + i) < width && x < height) {
            output[(y + i) * height + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize input matrix with some values
    for (int i = 0; i < width * height; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Device memory
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
              (height + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);

    transposeShared<<<grid, block>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // verify
    bool ok = true;
    for (int r = 0; r < height && ok; ++r) {
        for (int c = 0; c < width; ++c) {
            if (h_out[c * height + r] != h_in[r * width + c]) {
                ok = false;
                break;
            }
        }
    }

    std::cout << (ok ? "Transpose successful" : "Transpose failed") << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
