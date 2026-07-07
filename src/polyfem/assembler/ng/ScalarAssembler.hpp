#pragma once

#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Span.hpp>

namespace polyfem::assembler::ng
{

	struct BSRMatrixRef
	{
		int rows = 0;      //< Scalar row num.
		int cols = 0;      //< Scalar col num.
		int block_dim = 0; //< Block dimension. Can be 1, 2, or 3.

		Span<int> row_ptr;   //< CSR row ptr.
		Span<int> col_idx;   //< CSR col index.
		Span<double> values; //< CSR values.

		// Get block row num.
		int block_rows() const { return block_dim == 0 ? 0 : rows / block_dim; }
		// Get block col num,.
		int block_cols() const { return block_dim == 0 ? 0 : cols / block_dim; }
	};

	template <typename Kernel, typename ExecutionPolicy>
	class ScalarAssembler
	{
	public:
		void assemble(
			bool project_to_psd,
			double time,
			double dt,
			basis::ng::ElementBasesView solution_bases,
			basis::ng::ElementBasesView geom_bases,
			AssemblyCacheView assembly_cache,
			Span<const double> solution,
			Span<const double> solution_prev,
			double *scalar_out,
			ExecutionPolicy policy = {});
	};

} // namespace polyfem::assembler::ng
