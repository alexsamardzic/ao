# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
include(CMakeDependentOption)

project(torchao)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TORCHAO_BUILD_EXECUTORCH_OPS "Building torchao ops for ExecuTorch." OFF)
option(TORCHAO_BUILD_MPS_OPS "Building torchao MPS ops" OFF)
option(TORCHAO_BUILD_CPU_AARCH64 "Build torchao's CPU aarch64 kernels" OFF)
option(TORCHAO_BUILD_KLEIDIAI "Download, build, and link against Arm KleidiAI library (arm64 only)" OFF)

if(NOT TORCHAO_INCLUDE_DIRS)
  set(TORCHAO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if(NOT DEFINED TORCHAO_PARALLEL_BACKEND)
    set(TORCHAO_PARALLEL_BACKEND aten_openmp)
endif()

include(CMakePrintHelpers)

add_compile_options("-Wall" "-Werror" "-Wno-deprecated")

include(CMakePrintHelpers)
message("TORCHAO_INCLUDE_DIRS: ${TORCHAO_INCLUDE_DIRS}")
include_directories(${TORCHAO_INCLUDE_DIRS})


if(TORCHAO_BUILD_CPU_AARCH64)
    message(STATUS "Building with cpu/aarch64")
    add_compile_definitions(TORCHAO_BUILD_CPU_AARCH64)
    add_compile_definitions(TORCHAO_ENABLE_ARM_NEON_DOT)

    # Defines torchao_kernels_aarch64
    add_subdirectory(kernels/cpu/aarch64)

    if(TORCHAO_BUILD_KLEIDIAI)
        message(STATUS "Building with Arm KleidiAI library")
        add_compile_definitions(TORCHAO_ENABLE_KLEIDI)
    endif()
endif()

add_subdirectory(ops/linear_8bit_act_xbit_weight)
add_subdirectory(ops/embedding_xbit)

add_library(torchao_ops_aten SHARED)
target_link_libraries(
    torchao_ops_aten PRIVATE
    torchao_ops_linear_8bit_act_xbit_weight_aten
    torchao_ops_embedding_xbit_aten
)
if (TORCHAO_BUILD_MPS_OPS)
    message(STATUS "Building with MPS support")
    add_subdirectory(ops/mps)
    target_link_libraries(torchao_ops_aten PRIVATE torchao_ops_mps_aten)
endif()

install(
    TARGETS torchao_ops_aten
    EXPORT _targets
    DESTINATION lib
)
if(TORCHAO_BUILD_EXECUTORCH_OPS)
    add_library(torchao_ops_executorch STATIC)
    target_link_libraries(torchao_ops_executorch PRIVATE
        torchao_ops_linear_8bit_act_xbit_weight_executorch
        torchao_ops_embedding_xbit_executorch
    )
    install(
        TARGETS
            torchao_ops_executorch
            torchao_ops_linear_8bit_act_xbit_weight_executorch
            torchao_ops_embedding_xbit_executorch
        EXPORT _targets
        DESTINATION lib
    )
endif()
