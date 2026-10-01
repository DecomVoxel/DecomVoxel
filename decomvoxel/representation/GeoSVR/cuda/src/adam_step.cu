/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#include "adam_step.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace ADAM_STEP {

__forceinline__ __device__ float lerp(float v0, float v1, float t)
{
	// Compute (1-t) * v0 + t * v1
	return fmaf(t, v1, fmaf(-t, v0, v0));
}


template <bool sparse>
__global__ void unbiased_adam_step_cuda_kernel(
	const int numel,
	float* __restrict__ param,
	const float* __restrict__ grad,
	float* __restrict__ exp_avg,
	float* __restrict__ exp_avg_sq,
	const float step_size, const float beta1, const float beta2,
	const float rsqrt_bias_correction2, const float eps)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx >= numel)
		return;

	// Check gradient
	const float grad_val = grad[idx];
	if (sparse && grad_val == 0.0f)
		return;

	// Load parameters
	float exp_avg_val = exp_avg[idx];
	float exp_avg_sq_val = exp_avg_sq[idx];

	// Adam step
	// beta1 * exp_avg_val + (1.0f - beta1) * grad_val
	// beta2 * exp_avg_sq_val + (1.0f - beta2) * grad_val * grad_val
	exp_avg_val = lerp(grad_val, exp_avg_val, beta1); 
	exp_avg_sq_val = lerp(grad_val * grad_val, exp_avg_sq_val, beta2);

	const float denom = fmaf(sqrtf(exp_avg_sq_val), rsqrt_bias_correction2, eps);
	const float param_step = step_size * (exp_avg_val / denom);

	// Save back the new results
	param[idx] -= param_step;
	exp_avg[idx] = exp_avg_val;
	exp_avg_sq[idx] = exp_avg_sq_val;
}


template <bool sparse>
__global__ void biased_adam_step_cuda_kernel(
	const int numel,
	float* __restrict__ param,
	const float* __restrict__ grad,
	float* __restrict__ exp_avg,
	float* __restrict__ exp_avg_sq,
	const float lr, const float beta1, const float beta2, const float eps)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx >= numel)
		return;

	// Check gradient
	const float grad_val = grad[idx];
	if (sparse && grad_val == 0.0f)
		return;

	// Load parameters
	float exp_avg_val = exp_avg[idx];
	float exp_avg_sq_val = exp_avg_sq[idx];

	// Adam step
	// beta1 * exp_avg_val + (1.0f - beta1) * grad_val
	// beta2 * exp_avg_sq_val + (1.0f - beta2) * grad_val * grad_val
	exp_avg_val = lerp(grad_val, exp_avg_val, beta1); 
	exp_avg_sq_val = lerp(grad_val * grad_val, exp_avg_sq_val, beta2);

	const float denom = sqrtf(exp_avg_sq_val) + eps;
	const float param_step = lr * (exp_avg_val / denom);

	// Save back the new results
	param[idx] -= param_step;
	exp_avg[idx] = exp_avg_val;
	exp_avg_sq[idx] = exp_avg_sq_val;
}



void unbiased_adam_step(
	bool sparse,
	torch::Tensor& param,
	const torch::Tensor& grad,
	torch::Tensor& exp_avg,
	torch::Tensor& exp_avg_sq,
	const double step,
	const double lr, const double beta1, const double beta2, const float eps)
{
	const int numel = param.numel();

	const double bias_correction1 = 1.0 - pow(beta1, step);
	const double bias_correction2 = 1.0 - pow(beta2, step);

	const double step_size = lr / bias_correction1;

	const double rsqrt_bias_correction2 = rsqrt(bias_correction2);

	auto kernel_func = sparse ? unbiased_adam_step_cuda_kernel<true> :
								unbiased_adam_step_cuda_kernel<false>;

	kernel_func <<<(numel + 255) / 256, 256>>>(
		numel,
		param.contiguous().data_ptr<float>(),
		grad.contiguous().data_ptr<float>(),
		exp_avg.contiguous().data_ptr<float>(),
		exp_avg_sq.contiguous().data_ptr<float>(),
		step_size, beta1, beta2, rsqrt_bias_correction2, eps
	);
}

void biased_adam_step(
	bool sparse,
	torch::Tensor& param,
	const torch::Tensor& grad,
	torch::Tensor& exp_avg,
	torch::Tensor& exp_avg_sq,
	const float lr, const float beta1, const float beta2, const float eps)
{
	const int numel = param.numel();

	auto kernel_func = sparse ? biased_adam_step_cuda_kernel<true> :
								biased_adam_step_cuda_kernel<false>;

	kernel_func <<<(numel + 255) / 256, 256>>>(
		numel,
		param.contiguous().data_ptr<float>(),
		grad.contiguous().data_ptr<float>(),
		exp_avg.contiguous().data_ptr<float>(),
		exp_avg_sq.contiguous().data_ptr<float>(),
		lr, beta1, beta2, eps
	);
}

}
