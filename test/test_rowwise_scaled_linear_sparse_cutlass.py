import itertools
import random

import pytest
import torch
from torch.sparse import (
    SparseSemiStructuredTensor,
    to_sparse_semi_structured,
)
from torch.testing._internal.common_cuda import SM90OrLater

from torchao.dtypes import (
    Float8Layout,
    to_affine_quantized_floatx,
)
from torchao.ops import (
    rowwise_scaled_linear_sparse_cutlass_f8f8,
)

SparseSemiStructuredTensor._FORCE_CUTLASS = True


X_W_DTYPES = [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16)]
XQ_WQ_DTYPES = [
    (torch.float8_e5m2, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.float8_e4m3fn),
]
BATCH_SIZE = [1, 4, 8, 16, 32, 64]
SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]
USE_BIAS = [False, True]
BIAS_DTYPE = [torch.float16]
TEST_PARAMS = list(
    itertools.product(
        X_W_DTYPES,
        XQ_WQ_DTYPES,
        BATCH_SIZE,
        SIZE_MNK,
        USE_BIAS,
        BIAS_DTYPE,
    )
)


# FIXME: remove this!
X_W_DTYPES = [(torch.float16, torch.float16)]
XQ_WQ_DTYPES = [(torch.float8_e5m2, torch.float8_e4m3fn)]
BATCH_SIZE = [1]
SIZE_MNK = [(32, 64, 128)]
USE_BIAS = [True]
BIAS_DTYPE = [torch.float16]
TEST_PARAMS = list(
    itertools.product(
        X_W_DTYPES,
        XQ_WQ_DTYPES,
        BATCH_SIZE,
        SIZE_MNK,
        USE_BIAS,
        BIAS_DTYPE,
    )
)


def rand_sparse_semi_structured(r, c, dtype, device, choice=None):
    pattern = "2by4" if dtype != torch.float32 else "1by2"
    if pattern == "1by2":
        ksparse = 2
        choices = [[0, 1], [1, 0]]
    elif pattern == "2by4":
        ksparse = 4
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    assert c % ksparse == 0
    mask_entries = [choice or random.choice(choices) for i in range(r * c // ksparse)]
    mask = torch.tensor(mask_entries, dtype=torch.bool).view(r, c).to(device)
    dense = torch.randn(r, c, dtype=dtype, device=device)
    dense[dense == 0] = 1  # To prevent zeros except where mask applied.
    dense = dense.masked_fill(~mask, 0)
    return dense


def run_test_for_op(
    op,
    x_dtype,
    w_dtype,
    xq_dtype,
    wq_dtype,
    batch_size,
    size_mnk,
    use_bias,
    bias_dtype,
):
    # Ensure 2:4 sparsity is to be used for both original and
    # quantized sparse tensors, so that metadata stays the same
    # between the two.
    assert w_dtype in [torch.float16, torch.bfloat16]
    assert wq_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    size_m, size_n, size_k = size_mnk

    x = torch.randn((batch_size, size_m, size_k), dtype=x_dtype, device="cuda")
    w = rand_sparse_semi_structured(size_n, size_k, dtype=w_dtype, device="cuda")
    bias = torch.rand((size_n,), dtype=bias_dtype, device="cuda") if use_bias else None

    # FIXME: remove this!
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=x_dtype, device="cuda").view(8, 1).tile((size_m // 8, size_k))
    w = (w != 0).to(w_dtype)
    bias = torch.tensor([1, 2, 3, 4], dtype=x_dtype, device="cuda").tile((size_n // 4,))

    block_size = [1] * (x.dim() - 1) + [x.shape[-1]]
    x_aqt = to_affine_quantized_floatx(
        input_float=x,
        target_dtype=xq_dtype,
        block_size=block_size,
        _layout=Float8Layout(mm_config=None),
    )
    xq, xq_scales, zero_points = x_aqt.tensor_impl.get_plain()
    assert zero_points is None

    w_sp = to_sparse_semi_structured(w)
    block_size = [1] * (w_sp.packed.dim() - 1) + [w_sp.packed.shape[-1]]
    w_sp_packed_aqt = to_affine_quantized_floatx(
        input_float=w_sp.packed,
        target_dtype=wq_dtype,
        block_size=block_size,
        _layout=Float8Layout(mm_config=None),
    )
    wq, wq_scales, zero_points = w_sp_packed_aqt.tensor_impl.get_plain()
    assert zero_points is None
    wq_meta = w_sp.meta

    output_ref = torch.nn.functional.linear(
        x.to(torch.float32),
        w.to(torch.float32),
        bias.to(torch.float32) if bias is not None else bias,
    ).to(xq_scales.dtype)

    fn_inputs = (xq, xq_scales, wq, wq_meta, wq_scales, bias)
    try:
        output = op(*fn_inputs)
    except NotImplementedError:
        pytest.xfail("operator not implemented")

    torch.testing.assert_close(output, output_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not SM90OrLater, reason="FP8 is only supported on H100+ devices")
@pytest.mark.parametrize(
    "x_w_dtypes, xq_wq_dtypes, batch_size, size_mnk, use_bias, bias_dtype",
    TEST_PARAMS,
)
def test_rowwise_scaled_liner_sparse_cutlass_f8f8(
    x_w_dtypes,
    xq_wq_dtypes,
    batch_size,
    size_mnk,
    use_bias,
    bias_dtype,
):
    run_test_for_op(
        rowwise_scaled_linear_sparse_cutlass_f8f8,
        *x_w_dtypes,
        *xq_wq_dtypes,
        batch_size,
        size_mnk,
        use_bias,
        bias_dtype,
    )
