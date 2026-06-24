#pragma once

#include <polyfem/utils/Types.hpp>

#include <Eigen/SparseCore>

#include <vector>

namespace polyfem::assembler
{
	struct BlockCSRMatrix
	{
		int rows = 0;      //< Scalar row num.
		int cols = 0;      //< Scalar col num.
		int block_dim = 0; //< Block dimension. Can be 1, 2, or 3.

		std::vector<int> row_ptr;   //< CSR row ptr.
		std::vector<int> col_idx;   //< CSR col index.
		std::vector<double> values; //< CSR values.

		// Get block row num.
		int block_rows() const { return block_dim == 0 ? 0 : rows / block_dim; }
		// Get block col num,.
		int block_cols() const { return block_dim == 0 ? 0 : cols / block_dim; }
	};

	struct AssemblyResult
	{
		BlockCSRMatrix bsr;
		std::vector<Eigen::Triplet<double, int>> triplets;
	};
} // namespace polyfem::assembler
