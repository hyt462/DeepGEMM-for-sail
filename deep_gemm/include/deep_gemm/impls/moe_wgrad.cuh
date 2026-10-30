#pragma once

#include <cstdint>
#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>
#include <cute/atom/copy_traits_ppu0010_aiu.hpp>
#include <cute/atom/copy_traits_ppu0015_aiu.hpp>
#include <cute/algorithm/ppu_copy.hpp>

namespace deep_gemm {

// D[e,m,n] = sum_k A[offset[e]+k,m] * B[offset[e]+k,n] + C[e,m,n].
// A/B remain in token-major order in HBM. Cooperative loads transpose them
// into the same explicitly swizzled shared layout used by the fused forward
// kernel's A operand. No global transpose or per-expert host launch is needed.
// The initial implementation deliberately uses one synchronous shared stage.
template <typename DstT>
__device__ __forceinline__ void moe_wgrad_impl(
        const cutlass::bfloat16_t* a, const cutlass::bfloat16_t* b,
        DstT* d, const int32_t* counts, const DstT* c,
        int64_t m, int64_t n, int64_t experts) {
    using namespace cute;
    using SrcT = cutlass::bfloat16_t;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int Threads = 128;
#if __HGGC_ARCH__ == 100
    using ArchTag = cutlass::arch::PPU0010;
#else
    using ArchTag = cutlass::arch::PPU0015;
#endif
    using MmaInst = typename cutlass::gemm::config::GetMmaInst<
        ArchTag, SrcT, SrcT, float>::type;
    using TiledMma = TiledMMA<MMA_Atom<MmaInst>,
        Layout<Shape<_2, _2, _1>>>;
    using SmemAtom = decltype(composition(Swizzle<3, 3, 3>{},
        Layout<Shape<_8, Int<BK>>, Stride<Int<BK>, _1>>{}));
    using SmemLayoutA = decltype(tile_to_shape(SmemAtom{},
        Shape<Int<BM>, Int<BK>, _1>{}));
    using SmemLayoutB = decltype(tile_to_shape(SmemAtom{},
        Shape<Int<BN>, Int<BK>, _1>{}));

    extern __shared__ __align__(128) uint8_t moe_wgrad_smem[];
    auto sA = make_tensor(make_smem_ptr(reinterpret_cast<SrcT*>(moe_wgrad_smem)),
                          SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(reinterpret_cast<SrcT*>(moe_wgrad_smem)
                                      + BM * BK), SmemLayoutB{});
    TiledMma tiled_mma;
    const int thread = threadIdx.x;
    auto thr_mma = tiled_mma.get_thread_slice(thread);
    auto frag_a = thr_mma.partition_fragment_A(sA(_, _, 0));
    auto frag_b = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto accum = partition_fragment_C(tiled_mma, Shape<Int<BM>, Int<BN>>{});

#if __HGGC_ARCH__ >= 150
    using CopyAtomA = Copy_Atom<PPU0015_TSM_LD_SWZL<SrcT, BM, BK,
                                                false, false, 1>, SrcT>;
    using CopyAtomB = Copy_Atom<PPU0015_TSM_LD_SWZL<SrcT, BN, BK,
                                                true, false, 1>, SrcT>;
#else
    using CopyAtomA = Copy_Atom<PPU_U32x4_LDSM_N, SrcT>;
    using CopyAtomB = Copy_Atom<PPU_U32x4_LDSM_N, SrcT>;
#endif
    auto copy_a = make_tiled_copy_A(CopyAtomA{}, tiled_mma);
    auto copy_b = make_tiled_copy_B(CopyAtomB{}, tiled_mma);
#if __HGGC_ARCH__ >= 150
    // A TSM instruction is issued by the full warp from one common base.
    auto thread_copy_a = copy_a.get_thread_slice((thread / 32) * 32);
    auto thread_copy_b = copy_b.get_thread_slice((thread / 32) * 32);
    auto shared_a = thread_copy_a.partition_S(make_mix_tensor_like(sA));
    auto shared_b = thread_copy_b.partition_S(make_mix_tensor_like(sB));
#else
    auto thread_copy_a = copy_a.get_thread_slice(thread);
    auto thread_copy_b = copy_b.get_thread_slice(thread);
    auto shared_a = thread_copy_a.partition_S(sA);
    auto shared_b = thread_copy_b.partition_S(sB);
#endif
    auto registers_a = thread_copy_a.retile_D(frag_a);
    auto registers_b = thread_copy_b.retile_D(frag_b);
    constexpr int CopyK = decltype(size<2>(registers_a))::value;
    constexpr int AtomsPerCopy = decltype(size<2>(frag_a))::value / CopyK;
    CUTE_STATIC_ASSERT_V(size<2>(registers_a) == size<2>(registers_b));

    auto coords = make_identity_tensor(Shape<Int<BM>, Int<BN>>{});
    auto thread_coords = thr_mma.partition_C(coords);
    CUTE_STATIC_ASSERT_V(size(thread_coords) == size(accum));
    const int64_t tiles_m = (m + BM - 1) / BM;
    const int64_t tiles_n = (n + BN - 1) / BN;
    const int64_t tiles_per_expert = tiles_m * tiles_n;
    const int64_t tiles = experts * tiles_per_expert;
    __shared__ int64_t expert_begin;
    __shared__ int64_t expert_k;

    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int64_t expert = tile / tiles_per_expert;
        const int64_t tile_m = (tile % tiles_per_expert) / tiles_n;
        const int64_t tile_n = tile % tiles_n;
        if (thread == 0) {
            int64_t begin = 0;
            for (int64_t e = 0; e < expert; ++e) begin += counts[e];
            expert_begin = begin;
            expert_k = counts[expert];
        }
        __syncthreads();
        clear(accum);

        for (int64_t k0 = 0; k0 < expert_k; k0 += BK) {
            // Lane-adjacent global reads follow m/n, not the reduction axis.
            // Shared addressing performs the local transpose and bank swizzle.
            for (int idx = thread; idx < BK * BM; idx += Threads) {
                const int local_m = idx % BM;
                const int local_k = idx / BM;
                const int64_t gm = tile_m * BM + local_m;
                const int64_t gk = k0 + local_k;
                sA(local_m, local_k, 0) = (gm < m && gk < expert_k)
                    ? a[(expert_begin + gk) * m + gm] : SrcT(0.0f);
            }
            for (int idx = thread; idx < BK * BN; idx += Threads) {
                const int local_n = idx % BN;
                const int local_k = idx / BN;
                const int64_t gn = tile_n * BN + local_n;
                const int64_t gk = k0 + local_k;
                sB(local_n, local_k, 0) = (gn < n && gk < expert_k)
                    ? b[(expert_begin + gk) * n + gn] : SrcT(0.0f);
            }
            __syncthreads();
            auto shared_a_stage = shared_a(_, _, _, 0);
            auto shared_b_stage = shared_b(_, _, _, 0);
            for_each(make_int_sequence<CopyK>{}, [&](auto k_block) {
                copy(copy_a, shared_a_stage(_, _, k_block),
                     registers_a(_, _, k_block));
                copy(copy_b, shared_b_stage(_, _, k_block),
                     registers_b(_, _, k_block));
                CUTLASS_PRAGMA_UNROLL
                for (int atom = 0; atom < AtomsPerCopy; ++atom) {
                    auto k_atom = k_block * AtomsPerCopy + atom;
                    cute::gemm(tiled_mma, accum, frag_a(_, _, k_atom),
                               frag_b(_, _, k_atom), accum);
                }
            });
            // All warps finish reading this stage before any can overwrite it.
            __syncthreads();
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
            const int64_t gm = tile_m * BM + get<0>(thread_coords(i));
            const int64_t gn = tile_n * BN + get<1>(thread_coords(i));
            if (gm < m && gn < n) {
                const int64_t offset = (expert * m + gm) * n + gn;
                float value = accum(i);
                if (c != nullptr) value += static_cast<float>(c[offset]);
                // FP32 D never passes through BF16. C==D is safe: every output
                // element belongs to exactly one CTA/thread, with one store.
                d[offset] = DstT(value);
            }
        }
        __syncthreads();
    }
}
}  // namespace deep_gemm
