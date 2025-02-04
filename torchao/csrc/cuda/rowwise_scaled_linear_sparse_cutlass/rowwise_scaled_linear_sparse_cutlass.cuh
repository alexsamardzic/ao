#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/util/Exception.h>

#if defined(TORCHAO_USE_CUTLASS) && !defined(_WIN32) &&                   \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12020)
#define BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS
#endif

#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>

#include "cutlass_extensions/common.h"
#endif

#define OPERATOR_NAME "rowwise_scaled_linear_sparse_cutlass"

namespace torchao {

#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
template<
    typename XQDtype,
    typename WQDtype,
    typename YDtype,
    typename UseBias,
    typename BiasDtype,
    typename XScaleDtype,
    typename WScaleDtype,
    typename TileShape,
    typename ClusterShape>
void rowwise_scaled_linear_sparse_kernel_cutlass_sm9x(
  const at::Tensor& XQ, const at::Tensor& X_scale, const at::Tensor& WQ,
  const at::Tensor& W_meta, const at::Tensor& W_scale, const at::Tensor& bias,
  at::Tensor& Y) {
  // For CUTLASS, sparsified tensor must be the first operand, thus
  // the result will be calculated as:
  //    ((WQ @ XQ.T) * W_scale * X_scale.T + bias.T).T

  using SmArch = cutlass::arch::Sm90;

  // Use CUTLASS naming conventions for naming datatypes.
  using ElementA = WQDtype;
  using ElementB = XQDtype;
  using ElementD = YDtype;
  using ElementAScale = WScaleDtype;
  using ElementBScale = XScaleDtype;
  using ElementBias = BiasDtype;

  using LayoutTagA = cutlass::layout::RowMajor;
  using LayoutTagB = cutlass::layout::ColumnMajor;
  using LayoutTagD = cutlass::layout::ColumnMajor;

  constexpr auto AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr auto AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  constexpr auto AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // TODO: use different accumulator datatype if inputs are not float.
  using ElementAccumulator = float;
  using ElementCompute = float;

  using ProblemShape = cute::Shape<int, int, int, int>;

  using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using AScale =
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementAScale>;
  using ApplyAScale = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    Accum,
    AScale>;
  using BScale =
    cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementBScale>;
  using ApplyBScale = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    ApplyAScale,
    BScale>;
  using BiasScalar =
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementBias>;
  using BiasTensor =
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementBias>;
  using Bias = std::conditional_t<UseBias::value, BiasTensor, BiasScalar>;
  using ApplyBias = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus, ElementCompute, ElementCompute, RoundStyle>,
    ApplyBScale,
    Bias>;
  using EVT = ApplyBias;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      SmArch, cutlass::arch::OpClassSparseTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementD, LayoutTagD, AlignmentD,
      ElementD, LayoutTagD, AlignmentD,
      EpilogueSchedule,
      EVT>::CollectiveOp;
  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      SmArch, cutlass::arch::OpClassSparseTensorOp,
      ElementA, LayoutTagA, AlignmentA,
      ElementB, LayoutTagB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;


  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideE = StrideA;
  using ElementE = typename Gemm::GemmKernel::CollectiveMainloop::ElementE;
  using SparseConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;

  const int m = WQ.size(0);
  const int n = XQ.size(0);
  const int k = XQ.size(1);

  // FIXME: validate these checks.
  /*
  // Check for current CUTLASS limitations w.r.t. alignments.
  TORCH_CHECK(k % AlignmentA == 0, OPERATOR_NAME,
              " : Number of columns of tensor A must be divisible by ",
              AlignmentA);
  TORCH_CHECK(k % AlignmentB == 0, OPERATOR_NAME,
              " : Number of columns of tensor B must be divisible by ",
              AlignmentB);
  TORCH_CHECK(n % AlignmentD == 0, OPERATOR_NAME,
              " : Number of columns of tensor Y must be divisible by ",
              AlignmentD);
  */

  ProblemShape problem_shape(m, n, k, 1);
  const auto layout_A = SparseConfig::fill_layoutA(problem_shape);
  const auto layout_E = SparseConfig::fill_layoutE(problem_shape);
  const auto stride_B =
    cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  const auto stride_D =
    cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_shape,
    {
      (ElementA*)WQ.data_ptr(), layout_A, (ElementB*)XQ.data_ptr(), stride_B,
      (ElementE*)W_meta.data_ptr(), layout_E
    },
    {
      {},
      (ElementD*)Y.data_ptr(), stride_D, (ElementD*)Y.data_ptr(), stride_D
    }
  };

  const typename AScale::Arguments A_scale_arguments{
    (ElementAScale*)W_scale.data_ptr(),
    ElementAScale(1),
    {cute::_1{}, cute::_0{}, cute::_0{}}
  };
  const typename BScale::Arguments B_scale_arguments{
    (ElementBScale*)X_scale.data_ptr(),
    ElementBScale(1),
    {cute::_0{}, cute::_1{}, cute::_0{}}
  };
  const auto bias_arguments{
    [&]() -> typename Bias::Arguments {
      if constexpr (UseBias::value) {
        return {
          (ElementBias*)bias.data_ptr(),
          ElementBias(0),
          {cute::_1{}, cute::_0{}, cute::_0{}}
        };
      } else {
        return {ElementBias(0)};
      }
    }()
  };
  arguments.epilogue.thread = {
    {
      {
        {},                 // Accum
        A_scale_arguments,  // AScale
        {}                  // ApplyAScale
      },
      B_scale_arguments,    // TensorBScale
      {},                   // ApplyBScale
    },
    bias_arguments,         // Bias
    {}                      // ApplyBiass
  };

  Gemm gemm_op;

  cutlass::Status status;

  // Verify that GEMM operation with given arguments can be performed
  // by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Allocate workspace for CUTLASS mixed datatypes GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = XQ.new_empty({(int64_t)workspace_size},
                                      at::TensorOptions().dtype(at::kByte));

  // Initialize CUTLASS mixed datatypes GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  // Perform mixed datatypes GEMM operation.
  status = gemm_op.run(at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status, OPERATOR_NAME);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename XQDtype, typename WQDtype, typename... Types>
static void select_config(
    const at::Tensor& XQ, const at::Tensor& X_scale, const at::Tensor& WQ,
    const at::Tensor& W_meta, const at::Tensor& W_scale, const at::Tensor& bias,
    at::Tensor& Y) {
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const auto is_sm9x = dprops->major == 9;

  if (is_sm9x) {
    if constexpr ((std::is_same<XQDtype, cutlass::float_e4m3_t>::value &&
                   std::is_same<WQDtype, cutlass::float_e4m3_t>::value) ||
                  (std::is_same<XQDtype, cutlass::float_e5m2_t>::value &&
                   std::is_same<WQDtype, cutlass::float_e4m3_t>::value)) {
      // TODO: add some tuning
      using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
      using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
      rowwise_scaled_linear_sparse_kernel_cutlass_sm9x<
        XQDtype, WQDtype, Types..., TileShape, ClusterShape>(
          XQ, X_scale, WQ, W_meta, W_scale, bias, Y);
      return;
    }
  }

  TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported on SM", dprops->major, ".",
              dprops->minor, " for given operands");
}

template<typename XQDtype, typename WQDtype, typename YDtype, typename... Types>
static void
dispatch_on_bias(
    const at::Tensor& XQ, const at::Tensor& X_scale, const at::Tensor& WQ,
    const at::Tensor& W_meta, const at::Tensor& W_scale, const at::Tensor& bias,
    at::Tensor& Y) {
  if (bias.numel() == 0) {
    using UseBias = std::false_type;
    using BiasDtype = YDtype;
    select_config<XQDtype, WQDtype, YDtype, UseBias, BiasDtype, Types...>(
      XQ, X_scale, WQ, W_meta, W_scale, bias, Y);
    return;
  }

  using UseBias = std::true_type;
  if (bias.scalar_type() == at::ScalarType::Half) {
    using BiasDtype = cutlass::half_t;
    select_config<XQDtype, WQDtype, YDtype, UseBias, BiasDtype, Types...>(
      XQ, X_scale, WQ, W_meta, W_scale, bias, Y);
    return;
  } else if (bias.scalar_type() == at::ScalarType::BFloat16) {
    using BiasDtype = cutlass::bfloat16_t;
    select_config<XQDtype, WQDtype, YDtype, UseBias, BiasDtype, Types...>(
      XQ, X_scale, WQ, W_meta, W_scale, bias, Y);
    return;
  }

  TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported for datatype ", bias.scalar_type(),
              " for bias");
}

template<typename XQDtype, typename WQDtype, typename... Types>
 static void
dispatch_on_X_scale_and_W_scale(
    const at::Tensor& XQ, const at::Tensor& X_scale, const at::Tensor& WQ,
    const at::Tensor& W_meta, const at::Tensor& W_scale, const at::Tensor& bias,
    at::Tensor& Y) {
  TORCH_CHECK(Y.scalar_type() == X_scale.scalar_type(),
              OPERATOR_NAME, " : Operator not supported for Y datatype ",
              Y.scalar_type(), " as it's different from the first ",
              " operand scale datatype ", X_scale.scalar_type());

  if (X_scale.scalar_type() == at::ScalarType::Half &&
      W_scale.scalar_type() == at::ScalarType::Half) {
    using XScaleDtype = cutlass::half_t;
    using WScaleDtype = cutlass::half_t;
    using YDtype = cutlass::half_t;
    dispatch_on_bias<XQDtype, WQDtype, YDtype, XScaleDtype, WScaleDtype,
      Types...>(XQ, X_scale, WQ, W_meta, W_scale, bias, Y);
    return;
  } else if (X_scale.scalar_type() == at::ScalarType::BFloat16 &&
             W_scale.scalar_type() == at::ScalarType::BFloat16) {
    using XScaleDtype = cutlass::bfloat16_t;
    using WScaleDtype = cutlass::bfloat16_t;
    using YDtype = cutlass::bfloat16_t;
    dispatch_on_bias<XQDtype, WQDtype, YDtype, XScaleDtype, WScaleDtype,
      Types...>(XQ, X_scale, WQ, W_meta, W_scale, bias, Y);
    return;
  }

  TORCH_CHECK(false, OPERATOR_NAME,
              " : Operator not supported for combination of datatypes ",
              X_scale.scalar_type(), " for first operand scale and ",
              W_scale.scalar_type(), " for second operand scale");
}

template<typename XQDtype, typename WQDtype>
void
rowwise_scaled_linear_sparse_cutlass_check_inputs(
    const at::Tensor& XQ, const at::Tensor& X_scale, const at::Tensor& WQ,
    const at::Tensor& W_meta, const at::Tensor& W_scale,
    const at::Tensor& bias) {
  // Validate metadata datatype.
  TORCH_CHECK(W_meta.dtype() == at::kShort, OPERATOR_NAME,
              " : Expected WQ meta argument to be of torch.int16 datatype got ",
              WQ.dtype());

  // Validate layouts of arguments.
  TORCH_CHECK(XQ.dim() >= 2, OPERATOR_NAME,
              " : Expected XQ argument to be 2D or higher-dimensional tensor, "
              " got ", XQ.dim(), " dims");
  TORCH_CHECK(XQ.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected XQ argument to be strided, got layout ",
              XQ.layout());
  TORCH_CHECK(X_scale.dim() == XQ.dim() - 1, OPERATOR_NAME,
              " : Expected XQ scale argument to be ", XQ.dim() - 1,
              "D tensor, got ", X_scale.dim(), " dims");
  TORCH_CHECK(X_scale.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected XQ scale argument to be strided, got layout ",
              X_scale.layout());
  TORCH_CHECK(WQ.dim() == 2, OPERATOR_NAME,
              " : Expected WQ argument to be 2D tensor, got ", WQ.dim(),
              " dims");
  TORCH_CHECK(WQ.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected WQ argument to be strided, got layout ",
              WQ.layout());
  TORCH_CHECK(W_meta.dim() == 2, OPERATOR_NAME,
              " : Expected WQ meta argument to be 2D tensor, got ",
              W_meta.dim(), " dims");
  TORCH_CHECK(W_meta.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected WQ meta argument to be strided, got layout ",
              W_meta.layout());
  TORCH_CHECK(W_scale.dim() == 1 || W_scale.dim() == 2, OPERATOR_NAME,
              " : Expected WQ scale argument to be 1D or 2D tensor, ",
              "got ", W_scale.dim(), " dims");
  TORCH_CHECK(W_scale.layout() == at::Layout::Strided, OPERATOR_NAME,
              " : Expected WQ scale argument to be strided, got layout ",
              W_scale.layout());
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.dim() == 1, OPERATOR_NAME,
                " : Expected bias argument to be 1D tensor, got ", bias.dim(),
                " dims");
    TORCH_CHECK(bias.layout() == at::Layout::Strided, OPERATOR_NAME,
                " : Expected bias argument to be strided, got layout ",
                bias.layout());
  }

  // Validate sizes of arguments.
  const auto XQ_sizes = XQ.sizes().vec();
  TORCH_CHECK(XQ_sizes.back() == 2 * WQ.size(1), OPERATOR_NAME,
              " : Expected XQ argument to have ", 2 * WQ.size(1),
              " columns, but got ", XQ_sizes.back());
  const auto X_scale_sizes = X_scale.sizes().vec();
  for (auto i = 0; i < X_scale_sizes.size(); ++i)
    TORCH_CHECK(X_scale_sizes[i] == XQ_sizes[i], OPERATOR_NAME,
                " : Expected XQ scale argument size at position ", i, " to be ",
                XQ_sizes[i], ", but got ", X_scale_sizes[i]);
  TORCH_CHECK(WQ.size(1) % 16 == 0, OPERATOR_NAME,
              " : Expected WQ argument to have number of columns divisible by ",
              " 16, got ", WQ.size(1));
  TORCH_CHECK(W_meta.size(0) == WQ.size(0), OPERATOR_NAME,
              " : Expected WQ meta argument to have ", WQ.size(0),
              " rows, got ", W_meta.numel(), " rows");
  TORCH_CHECK(W_meta.size(1) * W_meta.element_size() == WQ.size(1) / 4,
              OPERATOR_NAME, " : Expected WQ meta argument to hold ",
              WQ.size(1) / 4, " bytes per row to encode sparsity of WQ "
              "argument, got ", W_meta.size(1) * W_meta.element_size(),
              " bytes");
  TORCH_CHECK(W_scale.numel() == WQ.size(0), OPERATOR_NAME,
              " : Expected WQ scale argument to have ", WQ.size(0),
              " elements, got ", W_scale.numel(), " elements");
  if (bias.numel() > 0) {
    TORCH_CHECK(bias.numel() == WQ.size(0), OPERATOR_NAME,
                " : Expected bias argument to have ", WQ.size(0),
                " elements, got ", bias.numel(), " elements");
  }

  // Validate strides of arguments.
  const auto XQ_strides = XQ.strides();
  TORCH_CHECK(XQ_strides[XQ_strides.size() - 1] == 1, OPERATOR_NAME,
              " : Expected XQ argument in row-major layout");
  auto XQ_stride_expected = XQ_strides[XQ_strides.size() - 2];
  for (int i = XQ_strides.size() - 3; i >= 0; --i) {
    XQ_stride_expected *= XQ_sizes[i + 1];
    TORCH_CHECK(XQ_strides[i] == XQ_stride_expected, OPERATOR_NAME,
                " : Expected XQ argument in row-major layout");
  }
  TORCH_CHECK(X_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected XQ scale argument to be contiguous");
  const auto WQ_strides = WQ.strides();
  TORCH_CHECK(WQ_strides[0] >= 1 && WQ_strides[1] == 1, OPERATOR_NAME,
              " : Expected WQ argument in row-major layout");
  const auto W_meta_strides = W_meta.strides();
  TORCH_CHECK(W_meta_strides[0] >= 1 && W_meta_strides[1] == 1, OPERATOR_NAME,
              " : Expected WQ meta argument in row-major layout");
  TORCH_CHECK(W_scale.is_contiguous(), OPERATOR_NAME,
              " : Expected WQ scale argument to be contiguous");
  if (bias.numel() > 0) {
    const auto bias_strides = bias.strides();
    TORCH_CHECK(bias_strides[0] == 1, OPERATOR_NAME,
                " : Expected bias argument to be contiguous");
  }
}
#endif

template <typename XQDtype, typename WQDtype>
at::Tensor
rowwise_scaled_linear_sparse_cutlass(
    const at::Tensor& XQ, const at::Tensor& X_scale, const at::Tensor& WQ,
    const at::Tensor& W_meta, const at::Tensor& W_scale,
    const at::Tensor& bias) {
#if defined(BUILD_ROWWISE_SCALED_LINEAR_SPARSE_CUTLASS)
  // Check inputs.
  rowwise_scaled_linear_sparse_cutlass_check_inputs<XQDtype, WQDtype>(
      XQ, X_scale, WQ, W_meta, W_scale, bias);

  // Squash the input tensors as appropriate.
  const auto XQ_sizes = XQ.sizes().vec();
  const auto XQ_2d = XQ.reshape({-1, XQ_sizes.back()});
  const auto X_scale_1d = X_scale.reshape({-1});
  const auto W_scale_1d = W_scale.reshape({-1});

  // Create result tensor.
  at::Tensor Y = X_scale.new_empty({XQ_2d.size(0), WQ.size(0)});

  // Dispatch to appropriate kernel template.
  dispatch_on_X_scale_and_W_scale<XQDtype, WQDtype>(
      XQ_2d, X_scale_1d, WQ, W_meta, W_scale_1d, bias, Y);

  // Reshape and return Y tensor.
  auto Y_sizes = XQ_sizes;
  Y_sizes.back() = WQ.size(0);
  return Y.reshape(Y_sizes);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, OPERATOR_NAME);
  return at::Tensor{};
#endif
}

}  // namespace torchao
