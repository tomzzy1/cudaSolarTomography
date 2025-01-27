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
 * File linesearch_gpu.h: Line search for GPU implementation.
 * NOTE: Included from lbfgs.cu, not to be used on its own!
 **/

#ifndef LINESEARCH_GPU_H
#define LINESEARCH_GPU_H

#include "cost_function.h"
#include "timer.h"
#include "lbfgs.h"

namespace gpu_lbfgs
{
	// Step, function value and directional derivative at the
	// starting point of the line search
	__device__ real phi_prime_0;

	// Current, previous and correction step
	// Correction step is (alpha_cur - alpha_old)
	__device__ real alpha_cur;
	__device__ real alpha_old;
	__device__ real alpha_correction;

	// Directional derivative at alpha
	__device__ real phi_prime_alpha;

	// Low and high search interval boundaries
	__device__ real alpha_low;
	__device__ real alpha_high;
	__device__ real phi_low;
	__device__ real phi_high;
	__device__ real phi_prime_low;
	__device__ real phi_prime_high;

	// Strong Wolfe line search kernels
	__global__ void strongWolfeInit();
	__global__ void strongWolfePhase1(bool second_iter);
	__global__ void strongWolfePhase2(size_t tries);
}

bool lbfgs::gpu_linesearch(real *d_x, real *d_z, real *d_fk, real *d_gk,
						   size_t &evals, const real *d_gkm1, real *d_fkm1,
						   lbfgs::status &stat, real *step, size_t maxEvals,
						   timer *timer_evals, timer *timer_linesearch,
						   real *d_tmp, int *d_status)
{
	using namespace gpu_lbfgs;
	const size_t NX = m_costFunction.getNumberOfUnknowns();

	real phi_prime_0;

	dispatch_dot(NX, &phi_prime_0, d_z, d_gk, false); // phi_prime_0 = z' * gk

	if (phi_prime_0 >= 0.0)
	{
		stat = lbfgs::LBFGS_LINE_SEARCH_FAILED;
		return false;
	}

	CudaSafeCall(cudaMemcpyToSymbol(gpu_lbfgs::phi_prime_0, &phi_prime_0, sizeof(real))); // phi_prime_0 = z' * gk

	real *d_alpha_correction, *d_phi_prime_alpha;
	CudaSafeCall(cudaGetSymbolAddress((void **)&d_alpha_correction, gpu_lbfgs::alpha_correction));
	CudaSafeCall(cudaGetSymbolAddress((void **)&d_phi_prime_alpha, gpu_lbfgs::phi_prime_alpha));

	const real zero = 0.0f;
	const real one = 1.0f;
	CudaSafeCall(cudaMemcpyToSymbol(gpu_lbfgs::alpha_cur, &one, sizeof(real)));
	CudaSafeCall(cudaMemcpyToSymbol(gpu_lbfgs::alpha_old, &zero, sizeof(real)));
	CudaSafeCall(cudaMemcpyToSymbol(gpu_lbfgs::alpha_correction, &one, sizeof(real)));

	bool second_iter = false;

	for (;;)
	{
		// go from (x + alpha_old * z)
		// to      (x + alpha     * z)

		// xk += (alpha - alpha_old) * z;

		dispatch_axpy(NX, d_x, d_x, d_z, d_alpha_correction, true);

#ifdef LBFGS_TIMING
		timer_linesearch->stop();
		timer_evals->start();
#endif

		m_costFunction.f_gradf(d_x, d_fk, d_gk);

		CudaCheckError();
		cudaDeviceSynchronize();

		++evals;

#ifdef LBFGS_TIMING
		timer_evals->stop();
		timer_linesearch->start();
#endif

		dispatch_dot(NX, d_phi_prime_alpha, d_z, d_gk, true); // phi_prime_alpha = z' * gk;

		strongWolfePhase1<<<1, 1>>>(second_iter);

		CudaCheckError();
		cudaDeviceSynchronize();

		int ret;
		CudaSafeCall(cudaMemcpy(&ret, d_status, sizeof(int), cudaMemcpyDeviceToHost));
		// printf("WolfePhase1 ret %d %s\n", ret, cudaGetErrorString(cudaGetLastError()));

		// If both Armijo and Strong Wolfe hold, we're done
		if (ret == 1)
		{
			return true;
		}

		if (evals >= maxEvals)
		{
			stat = lbfgs::LBFGS_REACHED_MAX_EVALS;
			return false;
		}

		// We've bracketed a viable minimum, go find it in phase 2
		if (ret == 2)
			break;

		// Coudln't find a viable minimum in the range [0, alpha_max=1e8]
		if (ret == 3)
		{
			stat = lbfgs::LBFGS_LINE_SEARCH_FAILED;
			return false;
		}

		second_iter = true;
	}

	// The minimum is now bracketed in [alpha_low, alpha_high]
	// Find it...
	size_t tries = 0;

	for (;;)
	{
		tries++;

		// go from (x + alpha_old * z)
		// to      (x + alpha     * z)

		// xk += (alpha - alpha_old) * z;
		dispatch_axpy(NX, d_x, d_x, d_z, d_alpha_correction, true);

#ifdef LBFGS_TIMING
		timer_linesearch->stop();
		timer_evals->start();
#endif

		m_costFunction.f_gradf(d_x, d_fk, d_gk);

		CudaCheckError();
		cudaDeviceSynchronize();

		++evals;

#ifdef LBFGS_TIMING
		timer_evals->stop();
		timer_linesearch->start();
#endif

		dispatch_dot(NX, d_tmp, d_z, d_gk, true); // tmp = phi_prime_j = z' * gk;

		strongWolfePhase2<<<1, 1>>>(tries);

		CudaCheckError();
		cudaDeviceSynchronize();

		int ret;
		CudaSafeCall(cudaMemcpy(&ret, d_status, sizeof(int), cudaMemcpyDeviceToHost));
		// printf("ret2 %d\n", ret);

		if (ret == 1)
		{
			// The Armijo and Strong Wolfe conditions hold
			return true;
		}

		if (ret == 2)
		{
			// The search interval has become too small
			stat = lbfgs::LBFGS_LINE_SEARCH_FAILED;
			return false;
		}

		if (evals >= maxEvals)
		{
			stat = lbfgs::LBFGS_REACHED_MAX_EVALS;
			return false;
		}
	}

	// We don't get here
}

// #define APPROXIMATE_WOLFE

namespace gpu_lbfgs
{
	__device__ bool sufficientDecrease(float phi_0_, float phi_prime_0_, float phi_alpha_, float phi_prime_alpha_, float alpha_cur_)
	{
#ifndef APPROXIMATE_WOLFE
		const float c1 = 1e-4f;
		return phi_alpha_ > phi_0_ + c1 * alpha_cur_ * phi_prime_0_;
#else
		const float c1 = 0.1f;
		return phi_prime_alpha_ > (2.0f * c1 - 1.0f) * phi_prime_0_;
#endif
	}
	__global__ void strongWolfePhase1(bool second_iter)
	{
		// printf("strongWolfePhase1\n");
		// const real c1 = 1e-4f;
		const real c2 = 0.9f;

		const real phi_alpha = fk;

		// const bool armijo_violated = (phi_alpha > fkm1 + c1 * alpha_cur * phi_prime_0 || (second_iter && phi_alpha >= fkm1));
		// const bool strong_wolfe = (fabsf(phi_prime_alpha) <= -c2 * phi_prime_0);
		const bool armijo_violated = (sufficientDecrease(fkm1, phi_prime_0, phi_alpha, phi_prime_alpha, alpha_cur) || (second_iter && phi_alpha >= fkm1));
		const bool strong_wolfe    = (fabsf(phi_prime_alpha) <= -c2 * phi_prime_0);

		// If both Armijo and Strong Wolfe hold, we're done
		if (!armijo_violated && strong_wolfe)
		{
			step = alpha_cur;
			status = 1;
			return;
		}

		// If Armijio condition is violated, we've bracketed a viable minimum
		// Interval is [alpha_0, alpha]
		if (armijo_violated)
		{
			// printf("armijo phi_prime_low %f phi_prime_high %f\n", phi_prime_low, phi_prime_high);
			// printf("armijo phi_low %f phi_high %f\n", phi_low, phi_high);
			// printf("armijo phi_prime_0 %f phi_prime_alpha %f\n", phi_prime_0, phi_prime_alpha);
			alpha_low = 0.0f;
			alpha_high = alpha_cur;
			phi_low = fkm1;
			phi_high = phi_alpha;
			phi_prime_low = phi_prime_0;
			phi_prime_high = phi_prime_alpha;

			status = 2;
		}
		// If the directional derivative at alpha is positive, we've bracketed a viable minimum
		// Interval is [alpha, alpha_0]
		else if (phi_prime_alpha >= 0)
		{
			// printf("phi_prime_alpha >= 0 phi_prime_low %f phi_prime_high %f\n", phi_prime_low, phi_prime_high);
			alpha_low = alpha_cur;
			alpha_high = 0.0f;
			phi_low = phi_alpha;
			phi_high = fkm1;
			phi_prime_low = phi_prime_alpha;
			phi_prime_high = phi_prime_0;

			status = 2;
		}

		if (status == 2)
		{
			// For details check the comment for the same code in phase 2
			// printf("alpha_old %f, alpha_cur %f, alpha_correction %f, alpha_low %f, alpha_high %f\n", alpha_old, alpha_cur, alpha_correction, alpha_low, alpha_high);
			alpha_old = alpha_cur;

			alpha_cur = 0.5f * (alpha_low + alpha_high);
			alpha_cur += (phi_high - phi_low) / (phi_prime_low - phi_prime_high);

			if (alpha_cur < fminf(alpha_low, alpha_high) || alpha_cur > fmaxf(alpha_low, alpha_high))
				alpha_cur = 0.5f * (alpha_low + alpha_high);

			alpha_correction = alpha_cur - alpha_old;

			// printf("alpha_old %f, alpha_cur %f, alpha_correction %f\n", alpha_old, alpha_cur, alpha_correction);

			return;
		}

		// Else look to the "right" of alpha for a viable minimum
		real alpha_new = alpha_cur + 4 * (alpha_cur - alpha_old);
		alpha_old = alpha_cur;
		alpha_cur = alpha_new;
		alpha_correction = alpha_cur - alpha_old;

		// No viable minimum found in the interval [0, 1e8]
		if (alpha_cur > 1e8f)
		{
			status = 3;
			return;
		}

		status = 0;
	}

	__global__ void strongWolfePhase2(size_t tries)
	{
		// printf("strongWolfePhase2 tries %d\n", tries);
		// const real c1 = 1e-4f;
		const real c2 = 0.9f;

		const size_t minTries = 10;

		const real phi_0 = fkm1;
		const real phi_j = fk;
		const real phi_prime_j = tmp;

		// const bool armijo_violated = (phi_j > phi_0 + c1 * alpha_cur * phi_prime_0 || phi_j >= phi_low);
		// const bool strong_wolfe = (fabsf(phi_prime_j) <= -c2 * phi_prime_0);

		const bool armijo_violated = (sufficientDecrease(phi_0, phi_prime_0, phi_j, phi_prime_j, alpha_cur) || phi_j >= phi_low);
		const bool strong_wolfe    = (fabsf(phi_prime_j) <= -c2 * phi_prime_0);

		if (!armijo_violated && strong_wolfe)
		{
			// The Armijo and Strong Wolfe conditions hold
			step = alpha_cur;
			status = 1;
			return;
		}
		else if (fabsf(alpha_high - alpha_low) < 1e-5f || tries > minTries)
		{
			// The search interval has become too small
			printf("interval too small\n");
			status = 2;
			return;
		}
		else if (armijo_violated)
		{
			alpha_high = alpha_cur;
			phi_high = phi_j;
			phi_prime_high = phi_prime_j;
		}
		else
		{
			if (tmp * (alpha_high - alpha_low) >= 0)
			{
				alpha_high = alpha_low;
				phi_high = phi_low;
				phi_prime_high = phi_prime_low;
			}

			alpha_low = alpha_cur;
			phi_low = phi_j;
			phi_prime_low = phi_prime_j;
		}

		// Quadratic interpolation:
		// Least-squares fit a parabola to (alpha_low, phi_low),
		// (alpha_high, phi_high) with gradients phi_prime_low and
		// phi_prime_high and select the minimum of that parabola as
		// the new alpha

		alpha_old = alpha_cur;

		alpha_cur = 0.5f * (alpha_low + alpha_high);
		alpha_cur += (phi_high - phi_low) / (phi_prime_low - phi_prime_high);

		if (isnan(alpha_cur))
		{
			printf("nan in wolfe phase 2\n");
			status = 2;
			return;
		}

		if (alpha_cur < fminf(alpha_low, alpha_high) || alpha_cur > fmaxf(alpha_low, alpha_high))
			alpha_cur = 0.5f * (alpha_low + alpha_high);

		alpha_correction = alpha_cur - alpha_old;

		status = 0;
	}
}

#endif // LINESEARCH_GPU_H
