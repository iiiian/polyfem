#include <polyfem/utils/BSRToStiffnessMatrix.hpp>

#include <cassert>
#include <vector>

namespace polyfem
{
	StiffnessMatrix bsr_to_stiffness_matrix(
		BSRMatrixView bsr,
		Span<const Eigen::Triplet<double>> triplets)
	{
		assert(bsr.block_dim > 0);
		assert(bsr.rows % bsr.block_dim == 0);
		assert(bsr.cols % bsr.block_dim == 0);

		int bd = bsr.block_dim;
		int block_size = bd * bd;

		std::vector<Eigen::Triplet<double>> entries;
		entries.reserve(bsr.values.size() + triplets.size());

		for (int br = 0; br < bsr.block_rows(); ++br)
		{
			for (int p = bsr.row_ptr[br]; p < bsr.row_ptr[br + 1]; ++p)
			{
				int bc = bsr.col_idx[p];
				const double *block = bsr.values.data() + p * block_size;

				for (int i = 0; i < bd; ++i)
				{
					for (int j = 0; j < bd; ++j)
					{
						double value = block[i * bd + j];
						if (value != 0.0)
						{
							entries.emplace_back(br * bd + i, bc * bd + j, value);
						}
					}
				}
			}
		}

		entries.insert(entries.end(), triplets.begin(), triplets.end());

		StiffnessMatrix out(bsr.rows, bsr.cols);
		out.setFromTriplets(entries.begin(), entries.end());
		return out;
	}
} // namespace polyfem
