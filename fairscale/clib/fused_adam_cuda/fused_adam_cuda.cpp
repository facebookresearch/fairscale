#include <torch/extension.h>

// CUDA forward declaration
void fused_adam_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists, float lr, float beta1, float beta2, float eps, float grad_scale, float optim_scale, at::Tensor& found_inf, int step, int mode, int bias_correction, float decay);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("adam", &fused_adam_cuda, "Multi tensor Adam optimized CUDA implementation.");
}
