#include <torch/library.h>

#include "rowwise_scaled_linear_sparse_cutlass.cuh"

namespace torchao {

at::Tensor
rowwise_scaled_linear_sparse_cutlass_f8f8(
    const at::Tensor& xq, const at::Tensor& x_scale, const at::Tensor& wq,
    const at::Tensor& w_meta, const at::Tensor& w_scale,
    const at::Tensor& bias) {
  // Validate input datatypes.
  TORCH_CHECK(
      (xq.dtype() == at::kFloat8_e5m2 && wq.dtype() == at::kFloat8_e4m3fn) ||
      (xq.dtype() == at::kFloat8_e4m3fn && wq.dtype() == at::kFloat8_e4m3fn),
      __func__, " : The input datatypes combination ", xq.dtype(),
      " for xq and ", wq.dtype(), " for wq is not supported");

  // Dispatch to appropriate kernel template.
  if (xq.dtype() == at::kFloat8_e5m2 && wq.dtype() == at::kFloat8_e4m3fn) {
    using ElementA = cutlass::float_e5m2_t;
    using ElementB = cutlass::float_e4m3_t;
    return rowwise_scaled_linear_sparse_cutlass<ElementA, ElementB>(
        xq, x_scale, wq, w_meta, w_scale, bias);
  } else if (xq.dtype() == at::kFloat8_e4m3fn &&
             wq.dtype() == at::kFloat8_e4m3fn) {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    return rowwise_scaled_linear_sparse_cutlass<ElementA, ElementB>(
        xq, x_scale, wq, w_meta, w_scale, bias);
  }

  return at::Tensor{};
}

TORCH_LIBRARY_IMPL(torchao, CUDA, m) {
  m.impl("torchao::rowwise_scaled_linear_sparse_cutlass_f8f8",
         &rowwise_scaled_linear_sparse_cutlass_f8f8);
}

}  // namespace torchao
