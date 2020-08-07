#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/TensorUtils.h"
// #include "ATen/Type.h"
#include "ATen/AccumulateType.h"
#include <THC/THCGeneral.h>
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

#include "type_shim.h"

typedef enum{
    ADAM_MODE_0   =0, // eps under square root
    ADAM_MODE_1   =1  // eps outside square root
} adamMode_t;

template <typename PARAM_T, typename GRAD_T>
__global__ void adam_cuda_kernel(
        PARAM_T* __restrict__ p,
        at::Half* __restrict__ p_copy, // For mixed precision training, pass NULL if not needed
        float* __restrict__ m,
        float* __restrict__ v,
        const GRAD_T * __restrict__ g,
        const float b1,
        const float b2,
        const float eps,
        const float grad_scale,
        const float step_size,
        const size_t tsize,
        adamMode_t mode,
        const float decay)
{
        //Assuming 2D grids and 2D blocks
        const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
        const int threadsPerBlock = blockDim.x * blockDim.y;
        const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
        const int i = (blockId * threadsPerBlock + threadIdInBlock);
        const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;

        for (int j = i; j < tsize; j+=totThreads) {
                float scaled_grad = g[j]/grad_scale;
                m[j] = b1*m[j] + (1-b1)*scaled_grad;
                v[j] = b2*v[j] + (1-b2)*scaled_grad*scaled_grad;
                float denom;
                if (mode == ADAM_MODE_0)
                    denom = sqrtf(v[j] + eps);
                else // Mode 1
                    denom = sqrtf(v[j]) + eps;
                float update = (m[j]/denom) + (decay*p[j]);
                p[j] = (PARAM_T)((float)p[j] - (step_size*update));
                if (p_copy != NULL) p_copy[j] = (at::Half) p[j];
        }
}

template <int DEPTH, typename PARAM_T, typename GRAD_T>
struct AdamFunctor
{
    __device__ __forceinline__ void operator()(
        int chunk_size,
        volatile int* noop_gmem,
        TensorListMetadata<DEPTH>& tl,
        const float b1,
        const float b2,
        const float eps,
        const float grad_scale,
        const float step_size,
        adamMode_t mode,
        const float decay)
    {
        int tensor_loc = tl.block_to_tensor[blockIdx.x];
        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.sizes[tensor_loc];

        PARAM_T* p = (PARAM_T *)tl.addresses[0][tensor_loc];
        p += chunk_idx*chunk_size;
        float* m = (float *)tl.addresses[1][tensor_loc];
        m += chunk_idx*chunk_size;
        float* v = (float *)tl.addresses[2][tensor_loc];
        v += chunk_idx*chunk_size;
        GRAD_T* g = (GRAD_T *)tl.addresses[3][tensor_loc];
        g += chunk_idx*chunk_size;
        at::Half* p_copy = NULL;
        if (DEPTH == 5) {
            p_copy = (at::Half*)tl.addresses[4][tensor_loc];
            p_copy += chunk_idx*chunk_size;
        }

        n -= chunk_idx*chunk_size;

        PARAM_T incoming_p[ILP];
        float incoming_m[ILP];
        float incoming_v[ILP];
        GRAD_T incoming_g[ILP];

        for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP) {

            #pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                incoming_p[ii] = 0;
                incoming_m[ii] = 0;
                incoming_v[ii] = 0;
                incoming_g[ii] = 0;

                int i = i_start + threadIdx.x + ii*blockDim.x;
                if (i < n && i < chunk_size) {
                    incoming_p[ii] = static_cast<PARAM_T>(p[i]);
                    incoming_m[ii] = m[i];
                    incoming_v[ii] = v[i];
                    incoming_g[ii] = static_cast<GRAD_T>(g[i]);
                }
            }

            // note for clarification to future michael:
            // From a pure memory dependency perspective, there's likely no point unrolling
            // the write loop, since writes just fire off once their LDGs arrive.
            // Put another way, the STGs are dependent on the LDGs, but not on each other.
            // There is still compute ILP benefit from unrolling the loop though.
            #pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                int j = i_start + threadIdx.x + ii*blockDim.x;

                if(j < n && j < chunk_size) {
                    float scaled_grad = incoming_g[ii]/grad_scale;
                    m[j] = b1*incoming_m[ii] + (1-b1)*scaled_grad;
                    v[j] = b2*incoming_v[ii] + (1-b2)*scaled_grad*scaled_grad;
                    float denom;
                    if (mode == ADAM_MODE_0)
                        denom = sqrtf(v[j] + eps);
                    else // Mode 1
                        denom = sqrtf(v[j]) + eps;
                    float update = (m[j]/denom) + (decay*incoming_p[ii]);
                    p[j] = (PARAM_T)(incoming_p[ii] - (step_size*update));
                    if (DEPTH == 5)  p_copy[j] = (at::Half) p[j];
                }
            }
        }
    }
};

void fused_adam_cuda(
        at::Tensor & p,
        at::Tensor & p_copy,
        at::Tensor & m,
        at::Tensor & v,
        at::Tensor & g,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float grad_scale,
        int step,
        int mode,
        int bias_correction,
        float decay)
{
    //Get tensor size
    int tsize = p.numel();
    //Determine #threads and #blocks
    const int threadsPerBlock = 512;
    const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p), "parameter tensor is too large to be indexed with int32");
    //Constants
    float step_size = 0;
    if (bias_correction == 1) {
        const float bias_correction1 = 1 - std::pow(beta1, step);
        const float bias_correction2 = 1 - std::pow(beta2, step);
        step_size = lr * std::sqrt(bias_correction2)/bias_correction1;
    }
    else {
        step_size = lr;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using namespace at; // prevents "toString is undefined" errors
    DISPATCH_FLOAT_AND_HALF(p.scalar_type(), 0, "adam_cuda_kernel",
        DISPATCH_FLOAT_AND_HALF(g.scalar_type(), 1, "adam_cuda_kernel",
            adam_cuda_kernel<scalar_t_0, scalar_t_1><<<blocks,threadsPerBlock, 0, stream>>>(
                p.DATA_PTR<scalar_t_0>(),
                p_copy.numel() ? p_copy.DATA_PTR<at::Half>() : NULL,
                m.DATA_PTR<float>(),
                v.DATA_PTR<float>(),
                g.DATA_PTR<scalar_t_1>(),
                beta1,
                beta2,
                eps,
                grad_scale,
                step_size,
                tsize,
                (adamMode_t) mode,
                decay
            );
        );
    );
    THCudaCheck(cudaGetLastError());
}

void fused_adam_cuda_mt(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, // p, m, v, g, p_copy
    float lr,
    float beta1,
    float beta2,
    float eps,
    float grad_scale,
    int step,
    int mode,
    int bias_correction,
    float decay) {

    //Constants
    float step_size = 0;
    if (bias_correction == 1) {
        const float bias_correction1 = 1 - std::pow(beta1, step);
        const float bias_correction2 = 1 - std::pow(beta2, step);
        step_size = lr * std::sqrt(bias_correction2)/bias_correction1;
    }
    else {
        step_size = lr;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    size_t tl_sz = tensor_lists.size();
    AT_ASSERTM(tl_sz == 4 || tl_sz == 5, "expected tensor lists of size 4 or 5");

    if (tl_sz == 4) {
        using namespace at; // prevents "toString is undefined" errors
        DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "adam_cuda_kernel",
            DISPATCH_FLOAT_AND_HALF(tensor_lists[3][0].scalar_type(), 1, "adam_cuda_kernel",
                multi_tensor_apply<4>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<4, scalar_t_0, scalar_t_1>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    (adamMode_t) mode,
                    decay
                );
            );
        );
    } else {
        using namespace at; // prevents "toString is undefined" errors
        DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "adam_cuda_kernel",
            DISPATCH_FLOAT_AND_HALF(tensor_lists[3][0].scalar_type(), 1, "adam_cuda_kernel",
                multi_tensor_apply<5>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<5, scalar_t_0, scalar_t_1>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    (adamMode_t) mode,
                    decay
                );
            );
        );
    }
    THCudaCheck(cudaGetLastError());
}
