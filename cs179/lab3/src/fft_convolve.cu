/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <math.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < padded_length) {
        out_data[idx].x = (raw_data[idx].x * impulse_v[idx].x - raw_data[idx].y * impulse_v[idx].y) / padded_length;
        out_data[idx].y = (raw_data[idx].x * impulse_v[idx].y + raw_data[idx].y * impulse_v[idx].x) / padded_length;
        idx += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

    extern __shared__ float shared_memory_data[];
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int shared_idx = threadIdx.x;
    int start_idx = blockIdx.x * blockDim.x;
    while (start_idx < padded_length) {
        // Initialize the shared memory
        shared_memory_data[shared_idx] = 0.0;
        __syncthreads();

        // Copy data from global memory to the shared memory
        if (global_idx < padded_length) {
            shared_memory_data[shared_idx] = out_data[global_idx].x > 0 ? out_data[global_idx].x : -out_data[global_idx].x;
        }
        __syncthreads();

        // Begin reduction
        int shared_idx_threshold = blockDim.x;
        for (int i = 0; i < (int)log2((double)blockDim.x); i++) {
            shared_idx_threshold /= 2;
            if (shared_idx < shared_idx_threshold) {
                shared_memory_data[shared_idx] = 
                    shared_memory_data[shared_idx] > shared_memory_data[shared_idx + shared_idx_threshold] ? shared_memory_data[shared_idx] : shared_memory_data[shared_idx + shared_idx_threshold];
            }
            __syncthreads();
        }

        if (shared_idx == 0) {
            // Use the maximum value inside this block to update the global maximum abs val
            atomicMax(max_abs_val, shared_memory_data[0]);
        }

        // Update the start_idx
        start_idx += blockDim.x * gridDim.x;
        global_idx += blockDim.x * gridDim.x;
    }

}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx > padded_length) {
        out_data[idx].x = out_data[idx].x / (*max_abs_val);
        idx += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<dim3(blocks), dim3(threadsPerBlock)>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<dim3(blocks), dim3(threadsPerBlock), threadsPerBlock * sizeof(float)>>>(out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<dim3(blocks), dim3(threadsPerBlock)>>>(out_data, max_abs_val, padded_length);
}
