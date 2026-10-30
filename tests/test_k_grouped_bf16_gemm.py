"""Standalone PPU K-grouped BF16 TN tests with CPU FP32 references.

Only deep_gemm.k_grouped_bf16_gemm_tn_contiguous is exercised. The optional
JSON destination is supplied by the caller; this script creates no repository
validation files by default. No other new MoE API or runner script is required.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import torch

METRICS, ENVIRONMENT = [], {}
ERRORS = (AssertionError, RuntimeError, ValueError, TypeError, NotImplementedError)


def reference(a, b, counts, c=None):
    out = torch.zeros((len(counts), a.shape[1], b.shape[1]), dtype=torch.float32)
    if c is not None:
        out.copy_(c.float())
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        out[expert] += a[start:end].float().T @ b[start:end].float()
        start = end
    assert start == a.shape[0] == b.shape[0]
    return out


@contextmanager
def forbid_python_host_reads():
    """Python tripwire, not a profiler claim about native runtime internals."""
    original_to = torch.Tensor.to
    def reject(*args, **kwargs):
        raise AssertionError("Trusted TN path read a tensor value on the host")
    def checked_to(value, *args, **kwargs):
        destination = kwargs.get("device")
        if args and isinstance(args[0], (str, torch.device, torch.Tensor)):
            destination = args[0].device if isinstance(args[0], torch.Tensor) else args[0]
        if destination is not None and torch.device(destination).type == "cpu":
            reject()
        return original_to(value, *args, **kwargs)
    with ExitStack() as stack:
        for name in ("cpu", "item", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__", "is_nonzero"):
            if hasattr(torch.Tensor, name):
                stack.enter_context(mock.patch.object(torch.Tensor, name, reject))
        stack.enter_context(mock.patch.object(torch.Tensor, "to", checked_to))
        for name in ("equal", "allclose", "is_nonzero"):
            stack.enter_context(mock.patch.object(torch, name, reject))
        yield


class KGroupedBF16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not getattr(torch.version, "sail", None):
            raise RuntimeError("PPU PyTorch is required; device tests were NOT run")
        accelerator = torch.accelerator.current_accelerator()
        if accelerator is None:
            raise RuntimeError("No PPU accelerator is visible")
        cls.device = torch.device(accelerator.type, torch.accelerator.current_device_index())
        cls.runtime = torch.get_device_module(cls.device)
        if not cls.runtime.is_available():
            raise RuntimeError("PPU runtime unavailable")
        cls.dg = importlib.import_module("deep_gemm")
        cls.fn = staticmethod(cls.dg.k_grouped_bf16_gemm_tn_contiguous)
        if cls.dg.k_grouped_bf16_gemm_tn_contiguous is cls.dg.unimplemented:
            raise RuntimeError("K-grouped TN is still an unimplemented placeholder")
        ENVIRONMENT.update(device=str(cls.device), device_name=cls.runtime.get_device_name(cls.device),
                           module_path=cls.dg.__file__, extension_path=cls.dg.deep_gemm_cpp.__file__)

    def setUp(self):
        torch.manual_seed(20261030)

    def layout(self, counts, psum=False):
        values = torch.tensor(counts, dtype=torch.int32, device=self.device)
        return values.cumsum(0, dtype=torch.int32) if psum else values

    def check(self, actual, expected, label, exact=False):
        self.runtime.synchronize()
        dtype = actual.dtype
        a, e = actual.detach().float().cpu(), expected.detach().to(dtype).float().cpu()
        self.assertEqual(a.shape, e.shape)
        self.assertTrue(bool(torch.isfinite(a).all()), label)
        delta = a - e
        rel = float(delta.norm() / e.norm().clamp_min(1e-30))
        rtol, atol = ((0., 0.) if exact else
                      ((3e-5, 2e-5) if dtype == torch.float32 else (.008, .002)))
        METRICS.append(dict(test=self._testMethodName, label=label, shape=list(a.shape), dtype=str(dtype),
                            max_abs=float(delta.abs().max()) if delta.numel() else 0.,
                            relative_l2=rel, rtol=rtol, atol=atol))
        torch.testing.assert_close(a, e, rtol=rtol, atol=atol, msg=label)
        self.assertLessEqual(rel, 0. if exact else (2e-5 if dtype == torch.float32 else .005), label)

    def test_dtype_layout_c_modes_uneven_and_long_k(self):
        for counts, m, n in (([0, 3, 5], 17, 35), ([0, 1, 33, 129, 0], 40, 24), ([0, 4097, 31], 32, 64)):
            a = (torch.randn(sum(counts), m) * .125).to(torch.bfloat16)
            b = (torch.randn(sum(counts), n) * .125).to(torch.bfloat16)
            ad, bd = a.to(self.device), b.to(self.device)
            for dtype in (torch.float32, torch.bfloat16):
                initial = (torch.randn(len(counts), m, n) * .2 + .125).to(dtype)
                for psum in (False, True):
                    for mode in ("overwrite", "inplace", "separate"):
                        with self.subTest(counts=counts, dtype=dtype, psum=psum, mode=mode):
                            out = torch.full(initial.shape, float("nan"), dtype=dtype, device=self.device)
                            c = None if mode == "overwrite" else initial.to(self.device)
                            if mode == "inplace":
                                out.copy_(c)
                                c = out
                            self.fn(ad, bd, out, None if psum else counts, self.layout(counts, psum), c,
                                    use_psum_layout=psum, validate_metadata=True)
                            self.check(out, reference(a, b, counts, None if c is None else initial), mode)
                            if mode == "separate":
                                self.check(c, initial, "independent C unchanged", exact=True)

    def test_nonzero_fp32_main_grad_two_different_microbatches(self):
        for projection, m, n in (("down", 40, 24), ("gate", 24, 40), ("up", 24, 40)):
            initial = torch.randn(4, m, n) * .2 + .125
            out, expected = initial.to(self.device), initial
            for step, counts in enumerate(([0, 1, 257, 3], [2, 0, 17, 1])):
                a = (torch.randn(sum(counts), m) * (.125 + .02 * step)).to(torch.bfloat16)
                b = (torch.randn(sum(counts), n) * .125 + .01 * step).to(torch.bfloat16)
                self.fn(a.to(self.device), b.to(self.device), out, None, self.layout(counts, True), out,
                        use_psum_layout=True)
                expected = reference(a, b, counts, expected)
                self.assertEqual(out.dtype, torch.float32)
                self.check(out, expected, "{} microbatch {}".format(projection, step + 1))
            self.assertTrue(bool((out.cpu() != out.cpu().to(torch.bfloat16).float()).any()))

    def test_empty_experts_total_k_zero_and_empty_dimensions(self):
        for counts, m, n in (([0, 0, 0], 17, 35), ([], 17, 35), ([0, 3, 0], 0, 8), ([2, 0], 8, 0)):
            for dtype in (torch.float32, torch.bfloat16):
                a = torch.empty((sum(counts), m), dtype=torch.bfloat16, device=self.device)
                b = torch.empty((sum(counts), n), dtype=torch.bfloat16, device=self.device)
                initial = torch.randn(len(counts), m, n).to(dtype)
                out = initial.to(self.device)
                self.fn(a, b, out, None, self.layout(counts, True), out, use_psum_layout=True)
                self.check(out, initial, "empty accumulation", exact=True)
                self.fn(a, b, out, counts, self.layout(counts))
                self.check(out, torch.zeros_like(initial), "empty overwrite", exact=True)

    def test_persistent_grid_over_4096_tiles(self):
        counts, m, n = [1] * 257, 256, 256  # 257 * 4 * 4 = 4112 tiles.
        a = (torch.randn(257, m) * .125).to(torch.bfloat16)
        b = (torch.randn(257, n) * .125).to(torch.bfloat16)
        ad, bd = a.to(self.device), b.to(self.device)
        product = reference(a, b, counts)
        out = torch.full(product.shape, float("nan"), device=self.device)
        self.fn(ad, bd, out, None, self.layout(counts))
        self.check(out, product, "persistent overwrite")
        out.fill_(.125)
        self.fn(ad, bd, out, None, self.layout(counts), out)
        self.check(out, product + .125, "persistent nonzero C=D")

    def test_counts_psum_equivalence_and_trusted_no_python_d2h(self):
        counts = [0, 3, 0, 8]
        a = torch.randn(11, 17, dtype=torch.bfloat16, device=self.device)
        b = torch.randn(11, 35, dtype=torch.bfloat16, device=self.device)
        first = torch.randn(4, 17, 35, device=self.device)
        second = first.clone()
        count_tensor, ends = self.layout(counts), self.layout(counts, True)
        self.fn(a, b, first, None, count_tensor, first, validate_metadata=False)
        self.fn(a, b, second, None, ends, second, use_psum_layout=True, validate_metadata=False)
        self.check(first, second, "counts/psum identical", exact=True)
        with forbid_python_host_reads():
            self.fn(a, b, first, None, count_tensor, first)
            self.fn(a, b, second, None, ends, second, use_psum_layout=True)
        self.check(first, second, "trusted counts/psum identical", exact=True)

    def test_structural_errors_and_aliases_rejected_before_writing(self):
        a = torch.zeros((4, 8), dtype=torch.bfloat16, device=self.device)
        b = torch.zeros((4, 16), dtype=torch.bfloat16, device=self.device)
        d = torch.full((2, 8, 16), .25, device=self.device)
        layout = self.layout([1, 3])
        calls = [lambda: self.fn(a.float(), b, d, None, layout),
                 lambda: self.fn(a, b[:3], d, None, layout),
                 lambda: self.fn(a, b, d, None, layout[:1]),
                 lambda: self.fn(a, b, d, None, layout.long()),
                 lambda: self.fn(a, b, d, None, layout, d.to(torch.bfloat16)),
                 lambda: self.fn(a, b, d, None, layout, compiled_dims="mm"),
                 lambda: self.fn(a, b, d, None, layout, validate_metadata="yes"),
                 lambda: self.fn(a, b, d, None, layout, epilogue="unsupported")]
        for call in calls:
            with self.assertRaises(ERRORS):
                call()
            self.check(d, torch.full_like(d, .25), "invalid structure preserves D", exact=True)
        storage = torch.full((d.numel() + 1,), .375, device=self.device)
        aliased_d, partial_c = storage[:-1].view_as(d), storage[1:].view_as(d)
        with self.assertRaises(ERRORS):
            self.fn(a, b, aliased_d, None, layout, partial_c)
        self.check(storage, torch.full_like(storage, .375), "partial C/D overlap", exact=True)
        shared = torch.full((d.numel(),), .5, dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ERRORS):
            self.fn(shared[:32].view_as(a), b, shared.view_as(d), None, layout)
        self.check(shared, torch.full_like(shared, .5), "A/D overlap", exact=True)

    def test_checked_bad_metadata_and_legacy_host_counts_preserve_output(self):
        a = torch.zeros((4, 8), dtype=torch.bfloat16, device=self.device)
        b = torch.zeros((4, 16), dtype=torch.bfloat16, device=self.device)
        d = torch.full((2, 8, 16), .25, device=self.device)
        cases = [(None, [-1, 5], False), (None, [1, 5], True), (None, [3, 2], True),
                 (None, [2147483647, -2147483643], True), ([1, 3], [2, 2], False),
                 ([True, 3], [1, 3], False), ([1, 2], [1, 3], False)]
        for ks, values, psum in cases:
            with self.subTest(ks=ks, values=values, psum=psum), self.assertRaises(ERRORS):
                self.fn(a, b, d, ks, torch.tensor(values, dtype=torch.int32, device=self.device), d,
                        use_psum_layout=psum, validate_metadata=True)
            self.check(d, torch.full_like(d, .25), "checked rejection preserves D", exact=True)

    def test_only_tn_added_and_existing_forward_exports_unchanged(self):
        for name in ("k_grouped_fp8_gemm_nt_contiguous", "k_grouped_fp8_gemm_tn_contiguous",
                     "m_grouped_fp8_gemm_nn_contiguous"):
            self.assertIs(getattr(self.dg, name), self.dg.unimplemented)
        for name in ("m_grouped_bf16_gemm_nt_contiguous", "m_grouped_bf16_gemm_nt_nopad"):
            self.assertIs(getattr(self.dg, name), getattr(self.dg.deep_gemm_cpp, name))
        for name in ("moe_bf16", "weighted_silu_and_mul", "moe_router_backward", "moe_scatter_backward",
                     "megatron_m_grouped_gemm", "megatron_k_grouped_gemm", "k_grouped_bf16_wgrad"):
            self.assertFalse(hasattr(self.dg, name), name)
        for name in ("_m_grouped_bf16_gemm_nn_native", "_moe_combine_backward_native",
                     "_moe_gather_backward_native", "_moe_scatter_backward_native",
                     "_moe_router_backward_native", "weighted_silu_and_mul_backward"):
            self.assertFalse(hasattr(self.dg.deep_gemm_cpp, name), name)
        if hasattr(self.dg, "m_grouped_bf16_gemm_nn_contiguous"):
            self.assertIs(self.dg.m_grouped_bf16_gemm_nn_contiguous, self.dg.unimplemented)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="append")
    parser.add_argument("--json-output", help="Optional result path, preferably outside the repository")
    args = parser.parse_args()
    torch.set_num_threads(1)
    names = unittest.defaultTestLoader.getTestCaseNames(KGroupedBF16Tests)
    if args.test and set(args.test) - set(names):
        parser.error("Unknown test: " + ", ".join(sorted(set(args.test) - set(names))))
    suite = unittest.TestSuite(KGroupedBF16Tests(name) for name in names if not args.test or name in args.test)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.json_output:
        target = Path(args.json_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        report = dict(success=result.wasSuccessful(), tests_run=result.testsRun,
                      elapsed_seconds=time.perf_counter() - started, mode="standalone_ppu_k_grouped_tn",
                      torch_version=torch.__version__, sail_version=str(getattr(torch.version, "sail", None)),
                      environment=ENVIRONMENT, test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      metrics=METRICS, failures=[(str(test), trace) for test, trace in result.failures],
                      errors=[(str(test), trace) for test, trace in result.errors])
        target.write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
