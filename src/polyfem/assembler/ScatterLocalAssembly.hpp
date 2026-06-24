#pragma once

#include <polyfem/assembler/AssemblyResult.hpp>
#include <polyfem/basis/ElementBases.hpp>

#include <Eigen/SparseCore>

#include <vector>

namespace polyfem::assembler
{
	/// @brief Scatter local stiffness matrix into global BSR and triplets.
	/// @param row_mapping_id Element local row basis id.
	/// @param col_mapping_id Element local col basis id.
	/// @param mapping DOF mapping.
	/// @param local_matrix Row-major local assembly matrix.
	/// @param value_dim Local assembly matrix row/col dimension. Can be 1/2/3.
	/// @param bsr Output BSR matrix. Assumes already allocated with correct size and sparsity.
	/// @param missing_triplets Fallback output triplet lists if BSR sparsity does not match.
	/// @param transpose_local_matrix If true, transpose local assembly matrix before scattering.
	void scatter_local_assembly(
		int row_mapping_id,
		int col_mapping_id,
		basis::ng::DofMappingStoreView mapping,
		const double *local_matrix,
		int value_dim,
		BlockCSRMatrix &bsr,
		std::vector<Eigen::Triplet<double, int>> &missing_triplets,
		bool transpose_local_matrix = false);
} // namespace polyfem::assembler
