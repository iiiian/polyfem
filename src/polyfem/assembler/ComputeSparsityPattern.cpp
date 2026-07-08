#include <polyfem/assembler/ComputeSparsityPattern.hpp>

#include <cassert>

namespace polyfem::assembler
{

	BSRSparsityPattern compute_sparsity_pattern(
		const AssemblyEssentialsView &bases,
		int node_num,
		int block_dim)
	{
		assert(node_num >= 0);
		assert(block_dim >= 1 && block_dim <= 3);

		BSRSparsityPattern pattern{
			node_num * block_dim,
			node_num * block_dim,
			block_dim,
			{}};

		for (const ElementDesc &element : bases.element_desc)
		{
			Range range = element.dof_mapping_range;
			assert(range);
			assert(range.num == element.basis_desc.basis_num);
			assert(range.offset + range.num <= bases.dof_mapping_store.mapping_desc.size());

			for (int i = 0; i < range.num; ++i)
			{
				auto row_ids = bases.dof_mapping_store.get_node_ids(range.offset + i);

				for (int j = 0; j < range.num; ++j)
				{
					auto col_ids = bases.dof_mapping_store.get_node_ids(range.offset + j);

					for (int row_node : row_ids)
					{
						for (int col_node : col_ids)
						{
							assert(0 <= row_node && row_node < node_num);
							assert(0 <= col_node && col_node < node_num);
							pattern.insert(row_node * block_dim, col_node * block_dim);
						}
					}
				}
			}
		}

		return pattern;
	}

} // namespace polyfem::assembler
