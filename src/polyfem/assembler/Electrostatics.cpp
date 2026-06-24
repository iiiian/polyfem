#include "Electrostatics.hpp"

namespace polyfem::assembler
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}
	} // namespace

	void Electrostatics::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 1);

		epsilon_.add_multimaterial(index, params, units.permittivity(), root_path);
	}

	template <int element_dim>
	void Electrostatics::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		assert(local_element_matrix.size() == 1);
		double res = 0;
		for (int q = 0; q < data.quad_num; ++q)
		{
			double x, y, z;
			data.fill_phy_position<element_dim>(q, x, y, z);
			double epsilon = epsilon_(x, y, z, data.time, data.element_id);
			auto gradi = data.gather_basis_grad_phy<element_dim>(data.row_local_basis_id, q);
			auto gradj = data.gather_basis_grad_phy<element_dim>(data.col_local_basis_id, q);
			res += epsilon * gradi.dot(gradj) * data.weighted_measure[q];
		}
		local_element_matrix[0] = res;
	}

	void Electrostatics::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
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

	void Electrostatics::compute_stress_grad_multiply_mat(const OptAssemblerData &data,
														  const Eigen::MatrixXd &mat,
														  Eigen::MatrixXd &stress,
														  Eigen::MatrixXd &result) const
	{
		stress = data.grad_u_i;
		result = mat;
	}

	void Electrostatics::compute_stiffness_value(const double t,
												 const assembler::ElementAssemblyValues &vals,
												 const Eigen::MatrixXd &local_pts,
												 const Eigen::MatrixXd &displacement,
												 Eigen::MatrixXd &tensor) const
	{
		const int dim = local_pts.cols();
		tensor.resize(local_pts.rows(), dim * dim);
		assert(displacement.cols() == 1);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			for (int i = 0, idx = 0; i < dim; i++)
				for (int j = 0; j < dim; j++)
				{
					tensor(p, idx) = delta(i, j);
					idx++;
				}
		}
	}

	std::map<std::string, Assembler::ParamFunc> Electrostatics::parameters() const
	{
		std::map<std::string, ParamFunc> res;

		res["epsilon"] = [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return epsilon_(p, t, e);
		};

		return res;
	}

	double Electrostatics::compute_stored_energy(
		const bool is_volume,
		const int n_basis,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &gbases,
		const AssemblyValsCache &cache,
		const double t,
		const Eigen::MatrixXd &solution)
	{
		StiffnessMatrix K;
		assemble(is_volume, n_basis, bases, gbases, cache, t, K);

		Eigen::MatrixXd energy = 0.5 * solution.transpose() * (K * solution);
		assert(energy.size() == 1);
		return energy(0);
	}

} // namespace polyfem::assembler
