/**
 *   ___ _   _ ___   _     _       ___ ___ ___ ___
 *  / __| | | |   \ /_\   | |  ___| _ ) __/ __/ __|
 * | (__| |_| | |) / _ \  | |_|___| _ \ _| (_ \__ \
 *  \___|\___/|___/_/ \_\ |____|  |___/_| \___|___/
 *                                               2012
 *     by Jens Wetzl           (jens.wetzl@fau.de)
 *    and Oliver Taubmann (oliver.taubmann@fau.de)
 *
 * This work is licensed under a Creative Commons
 * Attribution 3.0 Unported License. (CC-BY)
 * http://creativecommons.org/licenses/by/3.0/
 *
 * File lbfgs.cu: Implementation of class lbfgs (except cpu_lbfgs).
 *
 **/

#include "lbfgs.h"
#include "timer.h"

#include <iostream>
#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <fstream>
#include <sstream>
using namespace std;
#define LBFGS_VERBOSE


namespace gpu_lbfgs {

	// Variables

	__device__ real fkm1;
	__device__ real fk;
	__device__ real tmp;

	__device__ real alpha[HISTORY_SIZE];
	__device__ real rho  [HISTORY_SIZE];
	__device__ real H0;
	__device__ real step;
	__device__ real tmp2;
	__device__ int status;

	// Small helper kernels for scalar operations in device memory needed during updates.
	// What they're used for is documented by comments in the places they are executed.
	// *** Use with a single thread only! ***

	__global__ void update1   (real *alpha_out, const real *sDotZ, const real *rho, real *minusAlpha_out);       // first  update loop
	__global__ void update2   (real *alphaMinusBeta_out, const real *rho, const real *yDotZ, const real *alpha); // second update loop
	__global__ void update3   (real *rho_out, real *H0_out, const real *yDotS, const real *yDotY);               // after line search
}

// linesearch_gpu.h is no real header, it contains
// part of the implementation and must be included
// after the variables above have been declared.
#include "linesearch_gpu.h" 

lbfgs::lbfgs(cost_function& cf)
	: m_costFunction(cf)
	, m_maxIter(cf.m_maxIter)
	, m_maxEvals(std::numeric_limits<size_t>::max())
	, m_gradientEps(cf.m_gradientEps)

{
	CublasSafeCall( cublasCreate(&m_cublasHandle) );
}

lbfgs::~lbfgs()
{
	CublasSafeCall( cublasDestroy(m_cublasHandle) );
}

std::string lbfgs::statusToString(lbfgs::status stat)
{
	switch (stat)
	{
		case LBFGS_BELOW_GRADIENT_EPS:
			return "Below gradient epsilon";
		case LBFGS_REACHED_MAX_ITER:
			return "Reached maximum number of iterations";
		case LBFGS_REACHED_MAX_EVALS:
			return "Reached maximum number of function/gradient evaluations";
		case LBFGS_LINE_SEARCH_FAILED:
			return "Line search failed";
		default:
			return "Unknown status";
	}
}

lbfgs::status lbfgs::minimize(real *d_x)
{
	return gpu_lbfgs(d_x);
}

lbfgs::status lbfgs::minimize_with_host_x(real *h_x)
{
	 const size_t NX = m_costFunction.getNumberOfUnknowns();
	 real *d_x;
	 cudaMalloc((void**)&d_x, NX * sizeof(real));
	 cudaMemcpy(d_x, h_x, NX * sizeof(real), cudaMemcpyHostToDevice);

	 status ret = minimize(d_x);

	 cudaMemcpy(h_x, d_x, NX * sizeof(real), cudaMemcpyDeviceToHost);
	 cudaFree(d_x);

	 return ret;
}

lbfgs::status lbfgs::gpu_lbfgs(real *d_x)
{
#ifdef LBFGS_TIMING
	timer timer_total     ("GPU_LBFGS_total"     );
	timer timer_evals     ("GPU_LBFGS_evals"     );
	timer timer_updates   ("GPU_LBFGS_updates"   );
	timer timer_linesearch("GPU_LBFGS_linesearch");

	timer_total.start();
#endif

	using namespace gpu_lbfgs;
	const size_t NX = m_costFunction.getNumberOfUnknowns();

	real *d_fkm1, *d_fk;  // f_{k-1}, f_k, function values at x_{k-1} and x_k
	real *d_gkm1, *d_gk;  // g_{k-1}, g_k, gradients       at x_{k-1} and x_k
	real *d_z;            // z,            search direction
	real *d_H0;           // H_0,          initial inverse Hessian (diagonal, same value for all elements)

	real *d_step;         // step          current step length
	real *d_tmp, *d_tmp2; // tmp, tmp2     temporary storage for intermediate results
	int   *d_status;       // status        return code for communication device -> host

	// Ring buffers for history

	real *d_s;            // s,            history of solution updates
	real *d_y;            // y,            history of gradient updates
	real *d_alpha;        // alpha,        history of alphas (needed for z updates)
	real *d_rho;          // rho,          history of rhos   (needed for z updates)

	// Allocations

	CudaSafeCall( cudaMalloc(&d_gk,   NX * sizeof(real)) );
	CudaSafeCall( cudaMalloc(&d_gkm1, NX * sizeof(real)) );
	CudaSafeCall( cudaMalloc(&d_z,    NX * sizeof(real)) );

	CudaSafeCall( cudaMalloc(&d_s,    HISTORY_SIZE * NX * sizeof(real)) );
	CudaSafeCall( cudaMalloc(&d_y,    HISTORY_SIZE * NX * sizeof(real)) );

	// Addresses of global symbols

	CudaSafeCall( cudaGetSymbolAddress((void**)&d_fkm1,   gpu_lbfgs::fkm1  ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_fk,     gpu_lbfgs::fk    ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_tmp,    gpu_lbfgs::tmp   ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_tmp2,   gpu_lbfgs::tmp2  ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_H0,     gpu_lbfgs::H0    ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_alpha,  gpu_lbfgs::alpha ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_rho,    gpu_lbfgs::rho   ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_step,   gpu_lbfgs::step  ) );
	CudaSafeCall( cudaGetSymbolAddress((void**)&d_status, gpu_lbfgs::status) );

	// reset all global symbols to enable repetitive run
	cudaMemset(d_fkm1, 0, sizeof(real));
	cudaMemset(d_fk, 0, sizeof(real));
	cudaMemset(d_tmp, 0, sizeof(real));

	cudaMemset(d_alpha, 0, sizeof(real) * HISTORY_SIZE);
	cudaMemset(d_rho, 0, sizeof(real) * HISTORY_SIZE);
	cudaMemset(d_H0, 0, sizeof(real));
	cudaMemset(d_step, 0, sizeof(real));
	cudaMemset(d_tmp2, 0, sizeof(real));
	cudaMemset(d_status, 0, sizeof(int));

	// Initialize

#ifdef LBFGS_TIMING
	timer_evals.start();
#endif

	m_costFunction.f_gradf(d_x, d_fk, d_gk);

	CudaCheckError();
	cudaDeviceSynchronize();

#ifdef LBFGS_TIMING
	timer_evals.stop();
#endif

	size_t evals = 1;

	status stat = LBFGS_REACHED_MAX_ITER;

#ifdef LBFGS_VERBOSE
	std::cout << "lbfgs::gpu_lbfgs()" << std::endl;
#endif

	// H0 = 1.0f;
	const real one = 1.0f;
	CudaSafeCall( cudaMemcpy(d_H0, &one, sizeof(real), cudaMemcpyHostToDevice) );

	size_t it;

	for (it = 0; it < m_maxIter; ++it)
	{
#ifdef LBFGS_VERBOSE
		real  h_y;
		CudaSafeCall( cudaMemcpy(&h_y, d_fk, sizeof(real), cudaMemcpyDeviceToHost) );

		real gknorm2;
		dispatch_dot(NX, &gknorm2, d_gk, d_gk, false);

		fprintf(stderr, "f(x) = % 12e, ||grad||_2 = % 12e\n", h_y, std::sqrt(gknorm2));
#endif
		cur_f = h_y;
		cur_grad = std::sqrt(gknorm2);

		// Check for convergence
		// ---------------------

		real gkNormSquared;
		real xkNormSquared;

		dispatch_dot(NX, &xkNormSquared, d_x,  d_x,  false);
		dispatch_dot(NX, &gkNormSquared, d_gk, d_gk, false);

		if (gkNormSquared < (m_gradientEps * m_gradientEps) * std::max(xkNormSquared, static_cast<real>(1.0)))
		{
			stat = LBFGS_BELOW_GRADIENT_EPS;
			break;
		}

		// change to use projected gradient tolerance as in scipy
		

		// Find search direction
		// ---------------------

#ifdef LBFGS_TIMING
		timer_updates.start();
#endif
		
		real minusOne = -1.0f;
		if (it == 0)
		{
			minusOne = -1.0f * NX / 1000;
			printf("coefficient in iteration 1: %f\n", minusOne);
		}
			
		dispatch_scale(NX, d_z, d_gk, &minusOne, false); // z = -gk

		const size_t MAX_IDX = std::min<size_t>(it, HISTORY_SIZE);

		for (size_t i = 1; i <= MAX_IDX; ++i)
		{
			size_t idx = index(it - i);

			dispatch_dot(NX, d_tmp, d_s + idx * NX, d_z); // tmp = sDotZ

			// alpha = tmp * rho
			// tmp = -alpha
			update1<<<1, 1>>>(d_alpha + idx, d_tmp, d_rho + idx, d_tmp);

			CudaCheckError();
			cudaDeviceSynchronize();

			// z += tmp * y
			dispatch_axpy(NX, d_z, d_z, d_y + idx * NX, d_tmp);
		}

		dispatch_scale(NX, d_z, d_z, d_H0); // z = H0 * z

		for (size_t i = MAX_IDX; i > 0; --i)
		{
			size_t idx = index(it - i);

			dispatch_dot(NX, d_tmp, d_y + idx * NX, d_z); // tmp = yDotZ

			// beta = rho * tmp
			// tmp = alpha - beta
			update2<<<1, 1>>>(d_tmp, d_rho + idx, d_tmp, d_alpha + idx);

			CudaCheckError();
			cudaDeviceSynchronize();

			// z += tmp * s
			dispatch_axpy(NX, d_z, d_z, d_s + idx * NX, d_tmp);
		}

#ifdef LBFGS_TIMING
		timer_updates.stop();
		timer_linesearch.start();
#endif

		CudaSafeCall( cudaMemcpy(d_fkm1, d_fk, 1  * sizeof(real), cudaMemcpyDeviceToDevice) ); // fkm1 = fk;
		CudaSafeCall( cudaMemcpy(d_gkm1, d_gk, NX * sizeof(real), cudaMemcpyDeviceToDevice) ); // gkm1 = gk;

		timer *t_evals = NULL, *t_linesearch = NULL;
#ifdef LBFGS_TIMING
		t_evals = &timer_evals;
		t_linesearch = &timer_linesearch;
#endif

		// (line search defined in linesearch_gpu.h)
		if (!gpu_linesearch(d_x, d_z, d_fk, d_gk, evals, d_gkm1, d_fkm1, stat, d_step,
							m_maxEvals, t_evals, t_linesearch, d_tmp, d_status))
		{
			break;
		}

#ifdef LBFGS_TIMING
		timer_linesearch.stop();
		timer_updates.start();
#endif

		// Update s, y, rho and H_0
		// ------------------------

		// s   = x_k - x_{k-1} = step * z
		// y   = g_k - g_{k-1}
		// rho = 1 / (y^T s)
		// H_0 = (y^T s) / (y^T y)

		real *d_curS = d_s + index(it) * NX;
		real *d_curY = d_y + index(it) * NX;

		dispatch_scale(NX, d_curS, d_z,  d_step);                   // s = step * z
		dispatch_axpy (NX, d_curY, d_gk, d_gkm1, &minusOne, false); // y = gk - gkm1

		dispatch_dot(NX, d_tmp,  d_curY, d_curS); // tmp  = yDotS
		dispatch_dot(NX, d_tmp2, d_curY, d_curY); // tmp2 = yDotY

		// rho = 1 / tmp
		// if (tmp2 > 1e-5)
		//   H0 = tmp / tmp2
		update3<<<1, 1>>>(d_rho + index(it), d_H0, d_tmp, d_tmp2);

		CudaCheckError();
		cudaDeviceSynchronize();

#ifdef LBFGS_TIMING
		timer_updates.stop();
#endif
	}

	// Deallocations

	CudaSafeCall( cudaFree(d_gk)   );
	CudaSafeCall( cudaFree(d_gkm1) );
	CudaSafeCall( cudaFree(d_z)    );

	CudaSafeCall( cudaFree(d_s)    );
	CudaSafeCall( cudaFree(d_y)    );

#ifdef LBFGS_TIMING
	timer_total.stop();

	timer_total.saveMeasurement();
	timer_evals.saveMeasurement();
	timer_updates.saveMeasurement();
	timer_linesearch.saveMeasurement();
#endif

#ifdef LBFGS_VERBOSE
	std::cout << "Number of iterations: " << it << std::endl;
	std::cout << "Number of function/gradient evaluations: " << evals << std::endl;
	std::cout << "Reason for termination: " << statusToString(stat) << std::endl;
#endif
	cur_iters = it;
	cur_stat = statusToString(stat);

	return stat;
}

// Vector operations
// -----------------

void lbfgs::dispatch_axpy(const size_t n, float *d_dst, const float *d_y, const float *d_x, const float *a, bool aDevicePointer) const
{
	const cublasPointerMode_t mode = aDevicePointer ? CUBLAS_POINTER_MODE_DEVICE
													: CUBLAS_POINTER_MODE_HOST;

	CublasSafeCall( cublasSetPointerMode(m_cublasHandle, mode) );

	if (d_dst != d_y)
		CudaSafeCall( cudaMemcpy(d_dst, d_y, n * sizeof(float), cudaMemcpyDeviceToDevice) );

	CublasSafeCall( cublasSaxpy(m_cublasHandle, int(n), a, d_x, 1, d_dst, 1) );
}

void lbfgs::dispatch_scale(const size_t n, float *d_dst, const float *d_x, const float *a, bool aDevicePointer) const
{
	const cublasPointerMode_t mode = aDevicePointer ? CUBLAS_POINTER_MODE_DEVICE
													: CUBLAS_POINTER_MODE_HOST;

	CublasSafeCall( cublasSetPointerMode(m_cublasHandle, mode) );

	if (d_dst != d_x)
		CudaSafeCall( cudaMemcpy(d_dst, d_x, n * sizeof(float), cudaMemcpyDeviceToDevice) );

	CublasSafeCall( cublasSscal(m_cublasHandle, int(n), a, d_dst, 1) );
}


void lbfgs::dispatch_dot(const size_t n, float *dst, const float *d_x, const float *d_y, bool dstDevicePointer) const
{
	const cublasPointerMode_t mode = dstDevicePointer ? CUBLAS_POINTER_MODE_DEVICE
													  : CUBLAS_POINTER_MODE_HOST;

	CublasSafeCall( cublasSetPointerMode(m_cublasHandle, mode) );

	CublasSafeCall( cublasSdot(m_cublasHandle, int(n), d_x, 1, d_y, 1, dst) );
}

void lbfgs::dispatch_axpy(const size_t n, double *d_dst, const double *d_y, const double *d_x, const double *a, bool aDevicePointer) const
{
	const cublasPointerMode_t mode = aDevicePointer ? CUBLAS_POINTER_MODE_DEVICE
													: CUBLAS_POINTER_MODE_HOST;

	CublasSafeCall( cublasSetPointerMode(m_cublasHandle, mode) );

	if (d_dst != d_y)
		CudaSafeCall( cudaMemcpy(d_dst, d_y, n * sizeof(double), cudaMemcpyDeviceToDevice) );

	CublasSafeCall( cublasDaxpy(m_cublasHandle, int(n), a, d_x, 1, d_dst, 1) );
}

void lbfgs::dispatch_scale(const size_t n, double *d_dst, const double *d_x, const double *a, bool aDevicePointer) const
{
	const cublasPointerMode_t mode = aDevicePointer ? CUBLAS_POINTER_MODE_DEVICE
													: CUBLAS_POINTER_MODE_HOST;

	CublasSafeCall( cublasSetPointerMode(m_cublasHandle, mode) );

	if (d_dst != d_x)
		CudaSafeCall( cudaMemcpy(d_dst, d_x, n * sizeof(double), cudaMemcpyDeviceToDevice) );

	CublasSafeCall( cublasDscal(m_cublasHandle, int(n), a, d_dst, 1) );
}


void lbfgs::dispatch_dot(const size_t n, double *dst, const double *d_x, const double *d_y, bool dstDevicePointer) const
{
	const cublasPointerMode_t mode = dstDevicePointer ? CUBLAS_POINTER_MODE_DEVICE
													  : CUBLAS_POINTER_MODE_HOST;

	CublasSafeCall( cublasSetPointerMode(m_cublasHandle, mode) );

	CublasSafeCall( cublasDdot(m_cublasHandle, int(n), d_x, 1, d_y, 1, dst) );
}

// -----------------

// Device / kernel functions
// -------------------------

namespace gpu_lbfgs
{
	__global__ void update1(real *alpha_out, const real *sDotZ, const real *rho, real *minusAlpha_out)
	{
		*alpha_out      = *sDotZ * *rho;
		*minusAlpha_out = -*alpha_out;
	}

	__global__ void update2(real *alphaMinusBeta_out, const real *rho, const real *yDotZ, const real *alpha)
	{
		const real beta = *rho * *yDotZ;
		*alphaMinusBeta_out = *alpha - beta;
	}

	__global__ void update3(real *rho_out, real *H0_out, const real *yDotS, const real *yDotY)
	{
		*rho_out = 1.0f / *yDotS;

		if (*yDotY > 1e-5)
			*H0_out = *yDotS / *yDotY;
	}
}

// ------------------
