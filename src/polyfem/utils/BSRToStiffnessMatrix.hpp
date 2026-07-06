#pragma once

#include <polyfem/utils/BlockCSRMatrix.hpp>
#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/SparseCore>

namespace polyfem
{
	StiffnessMatrix bsr_to_stiffness_matrix(
		BSRMatrixView bsr,
		Span<const Eigen::Triplet<double>> triplets = {});
} // namespace polyfem
