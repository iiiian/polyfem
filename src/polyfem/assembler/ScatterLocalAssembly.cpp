#include "ScatterLocalAssembly.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace polyfem::assembler
{
	namespace
	{
		// CAS loop atomic add.
		// Should be replaced by std::atomic_ref once we upgrade to C++ 20.
		void atomic_add(double &target, double value)
		{
#if defined(_MSC_VER)
			// For MSVC
			static_assert(sizeof(double) == sizeof(__int64), "Expected double and __int64 to have the same size.");
			assert(reinterpret_cast<std::uintptr_t>(&target) % alignof(__int64) == 0);

			auto *target_bits = reinterpret_cast<volatile __int64 *>(&target);
			__int64 old_bits = _InterlockedCompareExchange64(target_bits, 0, 0);
			while (true)
			{
				double old_value;
				std::memcpy(&old_value, &old_bits, sizeof(old_value));

				const double new_value = old_value + value;
				__int64 new_bits;
				std::memcpy(&new_bits, &new_value, sizeof(new_bits));

				const __int64 observed = _InterlockedCompareExchange64(target_bits, new_bits, old_bits);
				if (observed == old_bits)
					break;

				old_bits = observed;
			}
#else
			// For gcc and clang
			double old_value;
			__atomic_load(&target, &old_value, __ATOMIC_RELAXED);
			while (true)
			{
				double new_value = old_value + value;
				if (__atomic_compare_exchange(
						&target,
						&old_value,
						&new_value,
						true,
						__ATOMIC_RELAXED,
						__ATOMIC_RELAXED))
				{
					break;
				}
			}
#endif
		}

		int find_bsr_block_slot(const BlockCSRMatrix &bsr, int block_row, int block_col)
		{
			assert(block_row >= 0 && block_row + 1 < bsr.row_ptr.size());
			int begin = bsr.row_ptr[block_row];
			int end = bsr.row_ptr[block_row + 1];
			for (int block_id = begin; block_id < end; ++block_id)
			{
				if (bsr.col_idx[block_id] == block_col)
				{
					return block_id;
				}
			}
			return -1;
		}
	} // namespace

	void scatter_local_assembly(
		int row_mapping_id,
		int col_mapping_id,
		basis::ng::DofMappingStoreView mapping,
		const double *local_matrix,
		int value_dim,
		BlockCSRMatrix &bsr,
		std::vector<Eigen::Triplet<double, int>> &missing_triplets,
		bool transpose_local_matrix)
	{
		assert(local_matrix != nullptr);
		assert(value_dim > 0);
		assert(value_dim == bsr.block_dim);

		auto &row_desc = mapping.mapping_desc[row_mapping_id];
		auto &col_desc = mapping.mapping_desc[col_mapping_id];
		auto row_global_ids = slice_by_range(mapping.node_ids, row_desc.id_and_weight_range);
		auto row_weights = slice_by_range(mapping.weights, row_desc.id_and_weight_range);
		auto col_global_ids = slice_by_range(mapping.node_ids, col_desc.id_and_weight_range);
		auto col_weights = slice_by_range(mapping.weights, col_desc.id_and_weight_range);

		assert(row_global_ids.size() == row_weights.size());
		assert(col_global_ids.size() == col_weights.size());

		// Non conforming mesh has so-called virtual nodes.
		// A virtual node = weighted sum of multiple real nodes.
		for (int row_node_id = 0; row_node_id < row_global_ids.size(); ++row_node_id)
		{
			for (int col_node_id = 0; col_node_id < col_global_ids.size(); ++col_node_id)
			{
				int global_row_node_id = row_global_ids[row_node_id];
				int global_col_node_id = col_global_ids[col_node_id];
				int block_slot = find_bsr_block_slot(bsr, global_row_node_id, global_col_node_id);
				double weight = row_weights[row_node_id] * col_weights[col_node_id];

				double *block_values = block_slot >= 0
										   ? bsr.values.data() + block_slot * value_dim * value_dim
										   : nullptr;

				for (int i = 0; i < value_dim; ++i)
				{
					for (int j = 0; j < value_dim; ++j)
					{
						double value = transpose_local_matrix
										   ? local_matrix[j * value_dim + i]
										   : local_matrix[i * value_dim + j];
						if (value == 0.0)
						{
							continue;
						}

						double weighted_value = weight * value;
						// Sparsity cache hit.
						if (block_values)
						{
							atomic_add(block_values[i * value_dim + j], weighted_value);
						}
						// Sparsity cache miss. Append to triplets.
						else
						{
							missing_triplets.emplace_back(
								global_row_node_id * value_dim + i,
								global_col_node_id * value_dim + j,
								weighted_value);
						}
					}
				}
			}
		}
	}
} // namespace polyfem::assembler
