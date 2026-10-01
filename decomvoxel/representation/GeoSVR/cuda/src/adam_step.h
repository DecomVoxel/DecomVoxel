/*************************************************************************
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*************************************************************************/

#ifndef ADAM_STEP_H_INCLUDED
#define ADAM_STEP_H_INCLUDED

#include <torch/extension.h>

namespace ADAM_STEP {

// Python interface to run adam optimization step.
void unbiased_adam_step(
	bool sparse,
	torch::Tensor& param,
	const torch::Tensor& grad,
	torch::Tensor& exp_avg,
	torch::Tensor& exp_avg_sq,
	const double step,
	const double lr, const double beta1, const double beta2, const float eps);

void biased_adam_step(
	bool sparse,
	torch::Tensor& param,
	const torch::Tensor& grad,
	torch::Tensor& exp_avg,
	torch::Tensor& exp_avg_sq,
	const float lr, const float beta1, const float beta2, const float eps);

}

#endif
