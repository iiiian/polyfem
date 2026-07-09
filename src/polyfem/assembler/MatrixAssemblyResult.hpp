#pragma once

#include <polyfem/utils/BlockCSRMatrix.hpp>

#include <Eigen/SparseCore>

#include <vector>

namespace polyfem::assembler
{
	struct MatrixAssemblyResult
	{
		explicit MatrixAssemblyResult(const BSRSparsityPattern &sparsity)
			: static_mat(sparsity)
		{
		}

		void reset(ExecutionPolicy policy = {})
		{
			static_mat.reset(policy);
			dynamic_mat.clear();
		}

		BSRMatrix static_mat;
		std::vector<Eigen::Triplet<double>> dynamic_mat;
	};
} // namespace polyfem::assembler
