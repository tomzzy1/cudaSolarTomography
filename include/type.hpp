#pragma once
#ifdef __CUDACC__
#include <cusparse.h>
#endif
#include <type_traits>

namespace cudaSolarTomography
{
    // for controlling the datatype used in reconstruction and solver
    using real = float;

    using Params_t = double;

#ifdef __CUDACC__
#if __cplusplus >= 201703L
    constexpr auto CUDA_REAL = std::is_same_v<real, float> ? CUDA_R_32F : CUDA_R_64F;
#else
    constexpr auto CUDA_REAL = std::is_same<real, float>::value ? CUDA_R_32F : CUDA_R_64F;
#endif
#endif

    // workaround for old cuda toolkit version
// #if (CUDA_TOOLKIT_VERSION < 12) && defined(__CUDACC__)
//     using cusparseConstDnVecDescr_t = const cusparseDnVecDescr_t;
//     using cusparseConstSpMatDescr_t = const cusparseSpMatDescr_t;
// #endif
}
