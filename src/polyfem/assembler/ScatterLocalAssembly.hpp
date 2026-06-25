#pragma once

#include <polyfem/assembler/AssemblyResult.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/span.hpp>

#include <Eigen/SparseCore>

#include <vector>

namespace polyfem::assembler
{
	/// @brief Scatter local stiffness matrix into global BSR and triplets.
	/// @param row_mapping_id Element local row basis id.
	/// @param col_mapping_id Element local col basis id.
	/// @param mapping DOF mapping.
	/// @param local_matrix Row-major local assembly matrix block.
	/// @param value_dim Local assembly matrix row/col dimension. Can be 1/2/3.
	/// @param bsr Output BSR matrix. Assumes already allocated with correct size and sparsity.
	/// @param missing_triplets Fallback output triplet lists if BSR sparsity does not match.
	void scatter_local_assembly_matrix(
		int row_mapping_id,
		int col_mapping_id,
		basis::ng::DofMappingStoreView mapping,
		const double *local_matrix,
		int value_dim,
		BlockCSRMatrix &bsr,
		std::vector<Eigen::Triplet<double, int>> &missing_triplets);

	/// @brief Scatter a local element vector into a global vector.
	void scatter_local_assembly_vector(
		const basis::ng::ElementDesc &element_desc,
		basis::ng::DofMappingStoreView mapping,
		const double *local_vector,
		int basis_num,
		int value_dim,
		span<double> global_vector);

	/// @brief Scatter a full local element matrix by dispatching each basis-pair block.
	void scatter_dense_element_matrix(
		const basis::ng::ElementDesc &element_desc,
		basis::ng::DofMappingStoreView mapping,
		const double *element_matrix,
		int basis_num,
		int value_dim,
		BlockCSRMatrix &bsr,
		std::vector<Eigen::Triplet<double, int>> &missing_triplets);
} // namespace polyfem::assembler
