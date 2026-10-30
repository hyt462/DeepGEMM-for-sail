"""BF16 K-grouped TN GEMM on PPU1.5 with FP32 matrix accumulation.

This module provides only k_grouped_bf16_gemm_tn_contiguous. Inputs remain in
contiguous [sum_K, feature] storage; all experts execute in one native launch.
Checked metadata validation synchronizes; trusted device metadata does not.
The explicit primitive writes a caller-provided output and has no autograd rule.
"""

from numbers import Integral
from typing import Optional, Sequence

import torch

from . import deep_gemm_cpp as _C


def _tensor(name, value, *, ndim, dtypes, device=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if value.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {value.dtype}")
    if value.device.type in ("cpu", "meta"):
        raise ValueError(f"{name} must be on a PPU device")
    if not getattr(torch.version, "sail", None):
        raise RuntimeError("K-grouped BF16 TN requires the PPU PyTorch build")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _overlaps(a, b):
    # All callers have validated contiguity; byte intervals are exact and also
    # catch differently typed views of one storage. Empty tensors never overlap.
    if a.device != b.device or not a.numel() or not b.numel():
        return False
    a_start, b_start = a.data_ptr(), b.data_ptr()
    return (a_start < b_start + b.numel() * b.element_size()
            and b_start < a_start + a.numel() * a.element_size())


def _check_output_alias(d, inputs):
    for name, tensor in inputs:
        if _overlaps(d, tensor):
            raise ValueError(f"d must not overlap {name}")


def _metadata_validation(value, default):
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError("validate_metadata must be bool or None")
    return value


def _gemm_dimensions(*tensors):
    # Native scheduling arguments use int32 dimensions even though flattened
    # addresses and reduction offsets are computed with int64.
    if any(size > 2**31 - 1 for tensor in tensors for size in tensor.shape):
        raise ValueError("GEMM dimensions must fit int32")


def _check_c(c, d):
    if c is None:
        return
    _tensor("c", c, ndim=d.ndim, dtypes=(d.dtype,), device=d.device)
    if c.shape != d.shape:
        raise ValueError("c must have the same shape as d")
    if _overlaps(c, d) and c.data_ptr() != d.data_ptr():
        raise ValueError("c and d may be identical views or disjoint, but must not partially overlap")


def _counts_from_psum(ends):
    # End offsets have no leading zero. Every operation stays on the device;
    # the result's length is known from tensor metadata, not tensor values.
    return torch.cat((ends[:1], ends[1:] - ends[:-1]))


def _options(compiled_dims, epilogue):
    if epilogue is not None:
        raise NotImplementedError("custom epilogues are not supported by these kernels")
    if (not isinstance(compiled_dims, str)
            or any(dim not in "mnk" for dim in compiled_dims)
            or len(set(compiled_dims)) != len(compiled_dims)):
        raise ValueError("compiled_dims must contain distinct letters from 'mnk'")


@torch.no_grad()
def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks_cpu: Optional[Sequence[int]] = None,
    grouped_layout: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
    use_psum_layout: bool = False,
    epilogue=None,
    *,
    validate_metadata: Optional[bool] = None,
) -> None:
    """Write d[e] = a[e].T @ b[e] + c[e] with a native K-grouped kernel.

    a=[sum(K_e),M] and b=[sum(K_e),N] are contiguous BF16 expert segments.
    d=[E,M,N] is contiguous BF16 or FP32. grouped_layout is int32[E] on the
    same PPU: per-expert K counts by default, or cumulative end offsets without
    a leading zero when use_psum_layout=True. Arbitrary K_e, including zero,
    are supported. Reduction tails are zero-filled in the on-chip tiles.

    c=None overwrites D; C=D accumulates; distinct C is read in the epilogue. C must
    match D's shape/dtype/device, and partial C/D overlap is rejected. With empty
    experts, D becomes zero when C=None, and C otherwise. D cannot overlap A/B
    or metadata. Inputs need not be padded to 128 along K: every supplied row is
    included in the reduction, so callers must zero or omit any padding rows.

    Each expert/output tile owns its full reduction and accumulates in FP32,
    then writes FP32 or BF16 D once. C=D is safe without atomic accumulation.
    ks_cpu may be None. validate_metadata=None preserves checked legacy calls
    when ks_cpu is supplied, and selects the no-host-read path otherwise.
    validate_metadata=True copies counts to CPU to verify nonnegativity, the
    total row count and optional ks_cpu equality. False trusts the device
    values: counts must be nonnegative and sum to sum_K; cumulative ends must
    be nondecreasing, start nonnegative and end at sum_K. Malformed trusted
    metadata is outside the contract and can cause invalid device accesses.
    Shape/dtype/device/alias checks always run before writing D. compiled_dims
    is accepted for source compatibility; custom epilogues are unsupported.
    """
    _options(compiled_dims, epilogue)
    if not isinstance(use_psum_layout, bool):
        raise TypeError("use_psum_layout must be bool")
    checked = _metadata_validation(validate_metadata, ks_cpu is not None)
    _tensor("a", a, ndim=2, dtypes=(torch.bfloat16,))
    _tensor("b", b, ndim=2, dtypes=(torch.bfloat16,), device=a.device)
    _tensor("d", d, ndim=3, dtypes=(torch.bfloat16, torch.float32), device=a.device)
    _tensor("grouped_layout", grouped_layout, ndim=1,
            dtypes=(torch.int32,), device=a.device)
    _gemm_dimensions(a, b, d, grouped_layout)
    experts, m, n = d.shape
    if a.shape[1] != m or b.shape[1] != n or a.shape[0] != b.shape[0]:
        raise ValueError("expected a=[sum_K,M], b=[sum_K,N], d=[E,M,N]")
    if grouped_layout.numel() != experts:
        raise ValueError("grouped_layout must contain one K count per expert")
    if experts == 0 and a.shape[0] != 0:
        raise ValueError("zero experts require zero input rows")
    _check_output_alias(d, (("a", a), ("b", b), ("grouped_layout", grouped_layout)))
    _check_c(c, d)
    host_counts = None
    if ks_cpu is not None:
        if isinstance(ks_cpu, (str, bytes, torch.Tensor)):
            raise TypeError("ks_cpu must be a host sequence of nonnegative integers or None")
        try:
            host_counts = list(ks_cpu)
        except TypeError as exc:
            raise TypeError("ks_cpu must be a host sequence of nonnegative integers or None") from exc
        if any(isinstance(k, bool) or not isinstance(k, Integral) for k in host_counts):
            raise TypeError("ks_cpu entries must be integers")
        host_counts = [int(k) for k in host_counts]
        if len(host_counts) != experts or any(k < 0 or k > 2**31 - 1 for k in host_counts):
            raise ValueError("ks_cpu must contain E nonnegative int32 K counts")
        if sum(host_counts) != a.shape[0]:
            raise ValueError("sum(ks_cpu) must equal the row count of a and b")
    counts = _counts_from_psum(grouped_layout) if use_psum_layout else grouped_layout
    if checked:
        actual = counts.detach().cpu().tolist()
        if any(k < 0 for k in actual) or sum(actual) != a.shape[0]:
            raise ValueError("grouped_layout must encode nonnegative counts summing to the input row count")
        if host_counts is not None and actual != host_counts:
            raise ValueError("grouped_layout must encode the same per-expert counts as ks_cpu")
    _C._k_grouped_bf16_gemm_tn_native(a, b, d, counts, c)
