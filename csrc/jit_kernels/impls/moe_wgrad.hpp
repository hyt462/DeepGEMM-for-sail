#pragma once

#include <algorithm>
#include <optional>
#include <torch/extension.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"

namespace deep_gemm {

struct MoeWgradLaunchArgs {
    const void* a;
    const void* b;
    void* d;
    const int32_t* counts;
    const void* c;
    int64_t m;
    int64_t n;
    int64_t experts;
    bool fp32_output;
    LaunchArgs launch_args;
};

class PPUMoeWgradRuntime final : public LaunchRuntime<PPUMoeWgradRuntime> {
public:
    using Args = MoeWgradLaunchArgs;
    static std::string generate_impl(const Args& args) {
        const std::string dst_type = args.fp32_output ? "float" : "cutlass::bfloat16_t";
        return std::string(R"(
#include <deep_gemm/impls/moe_wgrad.cuh>
using MoeWgradDst = )") + dst_type + R"(;
extern "C" __global__ __launch_bounds__(128)
void moe_wgrad(const void* a, const void* b, void* d, const int32_t* counts,
               const void* c, int64_t m, int64_t n, int64_t experts) {
    deep_gemm::moe_wgrad_impl<MoeWgradDst>(
        static_cast<const cutlass::bfloat16_t*>(a),
        static_cast<const cutlass::bfloat16_t*>(b),
        static_cast<MoeWgradDst*>(d), counts,
        static_cast<const MoeWgradDst*>(c), m, n, experts);
}
)";
    }
    static void launch_impl(const KernelHandle& kernel,
                            const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.a, args.b,
            args.d, args.counts, args.c, args.m, args.n, args.experts));
    }
};

// Shape/dtype/alias/device validation is performed by the public API wrapper.
static void moe_wgrad(const torch::Tensor& a, const torch::Tensor& b,
                      const torch::Tensor& d, const torch::Tensor& counts,
                      const std::optional<torch::Tensor>& c = std::nullopt) {
    const int64_t experts = d.size(0);
    const int64_t m = d.size(1);
    const int64_t n = d.size(2);
    if (experts == 0 || m == 0 || n == 0) return;
    const int64_t tiles = experts * ((m + 63) / 64) * ((n + 63) / 64);
    const int blocks = static_cast<int>(std::min<int64_t>(tiles, 4096));
    constexpr int shared_bytes = (64 + 64) * 64 * sizeof(uint16_t);
    MoeWgradLaunchArgs args{a.data_ptr(), b.data_ptr(), d.data_ptr(),
        counts.data_ptr<int32_t>(), c.has_value() ? c->data_ptr() : nullptr,
        m, n, experts, d.scalar_type() == torch::kFloat32,
        {dim3(blocks), dim3(128), shared_bytes}};
    const auto& runtime = compiler->build("moe_wgrad", PPUMoeWgradRuntime::generate(args));
    PPUMoeWgradRuntime::launch(runtime, args);
}

}  // namespace deep_gemm
