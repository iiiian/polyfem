#include "Stokes.hpp"

namespace polyfem::assembler
{
	StokesVelocity::StokesVelocity()
		: viscosity_("viscosity")
	{
	}
	void StokesVelocity::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() >= 1 && size() <= 3);

		viscosity_.add_multimaterial(index, params, units.viscosity(), root_path);
	}

	template <int element_dim>
	void StokesVelocity::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		// (gradi : gradj)  = <gradi, gradj> * Id

		assert(local_element_matrix.size() == size() * size());
		double dot = 0;
		for (int q = 0; q < data.quad_num; ++q)
		{
			double x, y, z;
			data.fill_phy_position<element_dim>(q, x, y, z);
			double viscosity = viscosity_(x, y, z, data.time, data.element_id);

			auto gradi = data.gather_basis_grad_phy<element_dim>(data.row_local_basis_id, q);
			auto gradj = data.gather_basis_grad_phy<element_dim>(data.col_local_basis_id, q);
			dot += gradi.dot(gradj) * data.weighted_measure[q] * viscosity;
		}

		for (int d = 0; d < size(); ++d)
			local_element_matrix[d * size() + d] = dot;
	}

	void StokesVelocity::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
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

	VectorNd StokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> val(size());
		for (int d = 0; d < size(); ++d)
			val(d) = pt(d).getValue();

		const auto nu = viscosity_(val, 0, 0);
		for (int d = 0; d < size(); ++d)
		{
			res(d) = nu * pt(d).getHessian().trace();
		}

		return res;
	}

	std::map<std::string, Assembler::ParamFunc> StokesVelocity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &nu = viscosity_;
		res["viscosity"] = [&nu](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return nu(p, t, e);
		};

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	StokesMixed::assemble(const MixedAssemblerData &data) const
	{
		// -(psii : div phij)  = psii * gradphij

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(rows() * cols());
		res.setZero();

		const Eigen::MatrixXd &psii = data.psi_vals.basis_values[data.i].val;
		const Eigen::MatrixXd &gradphij = data.phi_vals.basis_values[data.j].grad_t_m;
		assert(psii.size() == gradphij.rows());
		assert(gradphij.cols() == rows());

		for (int k = 0; k < gradphij.rows(); ++k)
		{
			res -= psii(k) * gradphij.row(k) * data.da(k);
		}

		return res;
	}
} // namespace polyfem::assembler
