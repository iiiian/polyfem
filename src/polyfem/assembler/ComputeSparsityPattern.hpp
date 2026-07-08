#pragma once

#include <polyfem/assembler/AssemblyEssentials.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>

namespace polyfem::assembler
{

	BSRSparsityPattern compute_sparsity_pattern(
		const AssemblyEssentialsView &bases,
		int node_num,
		int block_dim);

} // namespace polyfem::assembler
