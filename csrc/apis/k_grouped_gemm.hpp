#pragma once
#include <limits>
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "../jit_kernels/impls/moe_wgrad.hpp"

namespace deep_gemm::k_grouped_gemm {
// Internal binding: the public Python entry validates metadata values and
// output aliasing; structural/device checks below do not read device values.
static void check_tensor(const torch::Tensor& tensor, const torch::Tensor& device_ref,
                          int rank, at::ScalarType dtype) {
    TORCH_CHECK(!tensor.is_cpu() && !tensor.is_meta() && tensor.device() == device_ref.device() &&
        tensor.is_contiguous() && tensor.dim() == rank && tensor.scalar_type() == dtype,
        "K-grouped BF16 TN requires contiguous tensors with matching PPU device, rank and dtype");
    for (auto size : tensor.sizes())
        TORCH_CHECK(size <= std::numeric_limits<int>::max(), "dimension exceeds int32 indexing");
}
static void check_device(const torch::Tensor& tensor) {
    hggcDeviceProp props;
    DG_HGGC_RUNTIME_CHECK(hggcGetDeviceProperties(&props, tensor.get_device()));
    TORCH_CHECK(props.major == 8 && props.minor == 9, "K-grouped BF16 TN requires PPU1.5");
}
static void register_apis(pybind11::module_& m) {
    m.def("_k_grouped_bf16_gemm_tn_native", [](const torch::Tensor& a, const torch::Tensor& b,
            const torch::Tensor& d, const torch::Tensor& counts, const std::optional<torch::Tensor>& c) {
        check_tensor(a, a, 2, torch::kBFloat16);
        check_tensor(b, a, 2, torch::kBFloat16);
        TORCH_CHECK(d.scalar_type() == torch::kBFloat16 || d.scalar_type() == torch::kFloat32,
                    "wgrad output must be BF16 or FP32");
        check_tensor(d, a, 3, d.scalar_type());
        check_tensor(counts, a, 1, torch::kInt32);
        TORCH_CHECK(a.size(0) == b.size(0) && a.size(1) == d.size(1) &&
            b.size(1) == d.size(2) && counts.numel() == d.size(0), "invalid K-grouped TN shape");
        if (c) {
            check_tensor(*c, a, 3, d.scalar_type());
            TORCH_CHECK(c->sizes() == d.sizes(), "C and D must have identical shapes");
        }
        const c10::DeviceGuard guard(a.device());
        check_device(a);
        moe_wgrad(a, b, d, counts, c);
    }, py::arg("a"), py::arg("b"), py::arg("d"), py::arg("counts"), py::arg("c") = std::nullopt);
}
} // namespace deep_gemm::k_grouped_gemm
