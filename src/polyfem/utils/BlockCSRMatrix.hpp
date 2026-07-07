#pragma once

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#endif

namespace polyfem
{

	struct BSRSparsityPattern
	{
		int rows;      //< Scalar row num.
		int cols;      //< Scalar col num.
		int block_dim; //< Block dimension. Can be 1, 2, or 3.

		std::unordered_set<uint64_t> non_zeros;

		void insert(uint32_t row, uint32_t col);
		void join(const BSRSparsityPattern &other);
	};

	struct BSRMatrixMutableView
	{
		int rows;      //< Scalar row num.
		int cols;      //< Scalar col num.
		int block_dim; //< Block dimension. Can be 1, 2, or 3.

		Span<const int> row_ptr; //< CSR row ptr.
		Span<const int> col_idx; //< CSR col index.
		Span<double> values;     //< CSR values.

		// Get block row num.
		POLYFEM_BOTH int block_rows() const { return rows / block_dim; }
		// Get block col num,.
		POLYFEM_BOTH int block_cols() const { return cols / block_dim; }

		POLYFEM_BOTH double *get_block(int block_i, int block_j) const
		{
			int row_start = row_ptr[block_i];
			int row_end = row_ptr[block_i + 1];
			for (int i = row_start; i < row_end; ++i)
			{
				if (col_idx[i] == block_j)
				{
					return values.data() + block_dim * block_dim * i;
				}
			}
			return nullptr;
		}

		POLYFEM_BOTH double *get_entry(int i, int j) const
		{
			int block_i = i / block_dim;
			int block_j = j / block_dim;
			double *block_ptr = get_block(block_i, block_j);
			if (!block_ptr)
				return nullptr;

			return block_ptr + i * block_dim + j;
		}
	};

	class BSRMatrix
	{
	private:
		int rows_;
		int cols_;
		int block_dim_;
		int value_size_;

		std::vector<int> row_ptr_;
		std::vector<int> col_idx_;
		std::vector<double> values_;

#ifdef POLYFEM_WITH_CUDA
		bool need_host_device_sync_ = true;

		DeviceBuf<int> d_row_ptr_;
		DeviceBuf<int> d_col_idx_;
		DeviceBuf<double> d_values_;
#endif

	public:
		BSRMatrix(const BSRSparsityPattern &sparsity);

		/// Lazily allocate zero initialized value array and return matrix view.
		BSRMatrixMutableView view();

		/// Clear host value storage and all device storage. Keep topology data.
		void clear_storage();

#ifdef POLYFEM_WITH_CUDA
		/// Lazily copy topology to device, allocate zero initialized value array, and return device matrix view.
		BSRMatrixMutableView device_view(CudaExecutionPolicy policy = {});
#endif
	};
} // namespace polyfem
