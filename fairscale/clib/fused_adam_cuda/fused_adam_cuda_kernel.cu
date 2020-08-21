#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include "ATen/TensorUtils.h"
#include "ATen/AccumulateType.h"
#include <THC/THCGeneral.h>
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

typedef enum{
    ADAM_MODE_0   =0, // eps under square root
    ADAM_MODE_1   =1  // eps outside square root
} adamMode_t;

template <int DEPTH, typename PARAM_T, typename GRAD_T, typename OPTIM_T>
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
        const bool use_optim_scaling,
        const float optim_scale,
        float* found_inf_ptr,
        const float step_size,
        adamMode_t mode,
        const float decay)
    {
        int tensor_loc = tl.block_to_tensor[blockIdx.x];
        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.sizes[tensor_loc];

        PARAM_T* p = (PARAM_T *)tl.addresses[0][tensor_loc];
        p += chunk_idx*chunk_size;
        OPTIM_T* m = (OPTIM_T *)tl.addresses[1][tensor_loc];
        m += chunk_idx*chunk_size;
        OPTIM_T* v = (OPTIM_T *)tl.addresses[2][tensor_loc];
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
        OPTIM_T incoming_m[ILP];
        OPTIM_T incoming_v[ILP];
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
                    if (use_optim_scaling) {
                        // Optimizer state is in half precision and must be scaled
                        float scaled_grad = incoming_g[ii]/grad_scale;
                        float momentum = b1 * (incoming_m[ii] / optim_scale) + (1-b1)*scaled_grad;
                        float velocity = b2 * (incoming_v[ii] / optim_scale) + (1-b2)*scaled_grad*scaled_grad;

                        m[j] = static_cast<OPTIM_T>(momentum * optim_scale);
                        v[j] = static_cast<OPTIM_T>(velocity * optim_scale);

                        if (!isfinite(m[j]) || !isfinite(v[j])) {
                            *found_inf_ptr = 1.f;
                        }

                        float denom;
                        if (mode == ADAM_MODE_0)
                            denom = sqrtf(velocity + eps);
                        else // Mode 1
                            denom = sqrtf(velocity) + eps;
                        float update = (momentum/denom) + (decay*incoming_p[ii]);
                        p[j] = (PARAM_T)(incoming_p[ii] - (step_size*update));
                        if (DEPTH == 5)  p_copy[j] = (at::Half) p[j];
                    } else {
                        // Optimizer state is in floating point precision
                        float scaled_grad = incoming_g[ii]/grad_scale;
                        float momentum = b1 * incoming_m[ii] + (1-b1)*scaled_grad;
                        float velocity = b2 * incoming_v[ii] + (1-b2)*scaled_grad*scaled_grad;
                        m[j] = static_cast<OPTIM_T>(momentum);
                        v[j] = static_cast<OPTIM_T>(velocity);
                        float denom;
                        if (mode == ADAM_MODE_0)
                            denom = sqrtf(velocity + eps);
                        else // Mode 1
                            denom = sqrtf(velocity) + eps;
                        float update = (momentum/denom) + (decay*incoming_p[ii]);
                        p[j] = (PARAM_T)(incoming_p[ii] - (step_size*update));
                        if (DEPTH == 5)  p_copy[j] = (at::Half) p[j];
                    }
                }
            }
        }
    }
};

void fused_adam_cuda(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, // p, m, v, g, p_copy
    float lr,
    float beta1,
    float beta2,
    float eps,
    float grad_scale,
    float optim_scale,
    at::Tensor& found_inf,
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
    assert(tl_sz == 4 || tl_sz == 5);
    assert(tensor_lists[1][0].scalar_type() == tensor_lists[2][0].scalar_type());

    bool use_optim_scaling = (tensor_lists[1][0].scalar_type() == at::ScalarType::Half);
    float* found_inf_ptr = found_inf.data_ptr<float>();

    if(tl_sz == 5) {
        // Mixed precision case
        assert(tensor_lists[0][0].scalar_type() == at::ScalarType::Float);
        assert(tensor_lists[3][0].scalar_type() == at::ScalarType::Half);
        assert(tensor_lists[4][0].scalar_type() == at::ScalarType::Half);
        multi_tensor_apply<5>(
            BLOCK_SIZE,
            chunk_size,
            noop_flag,
            tensor_lists,
            AdamFunctor<5, float, at::Half, float>(),
            beta1,
            beta2,
            eps,
            grad_scale,
            use_optim_scaling,
            optim_scale,
            found_inf_ptr,
            step_size,
            (adamMode_t) mode,
            decay
        );
    } else {
        // tl_sz == 4
        assert(tensor_lists[0][0].scalar_type() == tensor_lists[3][0].scalar_type());
        if(tensor_lists[0][0].scalar_type() == at::ScalarType::Float) {
            // Full precision case
            assert(tensor_lists[1][0].scalar_type() == at::ScalarType::Float);
            multi_tensor_apply<4>(
                BLOCK_SIZE,
                chunk_size,
                noop_flag,
                tensor_lists,
                AdamFunctor<4, float, float, float>(),
                beta1,
                beta2,
                eps,
                grad_scale,
                use_optim_scaling,
                optim_scale,
                found_inf_ptr,
                step_size,
                (adamMode_t) mode,
                decay
            );
        } else if (tensor_lists[0][0].scalar_type() == at::ScalarType::Half) {
            if(tensor_lists[1][0].scalar_type() == at::ScalarType::Float) {
                // Memory-efficient mixed-precision case
                // ie FP16 model parameters and gradients; FP32 optimizer state
                multi_tensor_apply<4>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<4, at::Half, at::Half, float>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    use_optim_scaling,
                    optim_scale,
                    found_inf_ptr,
                    step_size,
                    (adamMode_t) mode,
                    decay
                );
            } else if (tensor_lists[1][0].scalar_type() == at::ScalarType::Half) {
                // Pure FP16 case
                multi_tensor_apply<4>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<4, at::Half, at::Half, at::Half>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    use_optim_scaling,
                    optim_scale,
                    found_inf_ptr,
                    step_size,
                    (adamMode_t) mode,
                    decay
                );
            } else {
                throw "Optimizer state must be of type float or half";
            }
        } else {
            throw "Parameters must be of type float or half";
        }
    }
    THCudaCheck(cudaGetLastError());
}
