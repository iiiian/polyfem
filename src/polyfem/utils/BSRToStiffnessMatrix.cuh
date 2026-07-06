#pragma once

#include <polyfem/utils/BSRToStiffnessMatrix.hpp>
#include <polyfem/utils/CUDAExecutionPolicy.hpp>

namespace polyfem
{
	StiffnessMatrix bsr_to_stiffness_matrix_device(
		BSRMatrixView bsr,
		Span<const Eigen::Triplet<double>> triplets = {},
		CudaExecutionPolicy policy = {});
} // namespace polyfem
