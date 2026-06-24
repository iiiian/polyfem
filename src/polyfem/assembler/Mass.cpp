#include "Mass.hpp"

namespace polyfem::assembler
{
	template <int element_dim>
	void Mass::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		assert(local_element_matrix.size() == size() * size());
		double tmp = 0;

		for (int q = 0; q < data.quad_num; ++q)
		{
			double x, y, z;
			data.fill_phy_position<element_dim>(q, x, y, z);
			const double rho = density_(0.0, 0.0, 0.0, x, y, z, data.time, data.element_id);
			tmp += rho
				   * data.gather_basis_value(data.row_local_basis_id, q)
				   * data.gather_basis_value(data.col_local_basis_id, q)
				   * data.weighted_measure[q];
		}

		for (int i = 0; i < size(); ++i)
			local_element_matrix[i * size() + i] = tmp;
	}

	void Mass::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		switch (data.elem_dim)
		{
		case 1:
			assemble_element_impl<1>(data, local_element_matrix);
			break;
		case 2:
			assemble_element_impl<2>(data, local_element_matrix);
			break;
		case 3:
			assemble_element_impl<3>(data, local_element_matrix);
			break;
		default:
			assert(false);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> Mass::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(false);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> result;

		return result;
	}

	void Mass::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size_ == 1 || size_ == 2 || size_ == 3);

		density_.add_multimaterial(index, params, units.density(), root_path);
	}

	std::map<std::string, Assembler::ParamFunc> Mass::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		res["rho"] = [this](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) { return this->density_(uv, p, t, e); };

		return res;
	}

	template <int element_dim>
	void HRZMass::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		assert(local_element_matrix.size() == size() * size());

		if (data.row_local_basis_id != data.col_local_basis_id)
			return;

		double sum_all_entries = 0;
		double sum_all_diag_entries = 0;
		double sum_target_diag_entries = 0;

		for (int i = 0; i < data.basis_num; ++i)
		{
			for (int j = 0; j < data.basis_num; ++j)
			{
				double entry = 0;
				for (int q = 0; q < data.quad_num; ++q)
				{
					entry += data.gather_basis_value(i, q) * data.gather_basis_value(j, q) * data.weighted_measure[q];
				}
				sum_all_entries += entry;
				if (i == j)
				{
					sum_all_diag_entries += entry;
					if (i == data.row_local_basis_id)
					{
						sum_target_diag_entries += entry;
					}
				}
			}
		}

		for (int i = 0; i < size(); ++i)
			local_element_matrix[i * size() + i] = sum_all_entries / sum_all_diag_entries * sum_target_diag_entries;
	}

	void HRZMass::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		switch (data.elem_dim)
		{
		case 1:
			assemble_element_impl<1>(data, local_element_matrix);
			break;
		case 2:
			assemble_element_impl<2>(data, local_element_matrix);
			break;
		case 3:
			assemble_element_impl<3>(data, local_element_matrix);
			break;
		default:
			assert(false);
		}
	}

} // namespace polyfem::assembler
