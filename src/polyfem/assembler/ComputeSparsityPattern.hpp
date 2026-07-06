#pragma once

#include <polyfem/assembler/ElementBases.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>

namespace polyfem::assembler
{

	BSRSparsityPattern compute_sparsity_pattern(
		const ElementBasesView &bases,
		int node_num,
		int block_dim);

} // namespace polyfem::assembler
