#include "Helmholtz.hpp"
#include <polyfem/utils/Bessel.hpp>

namespace polyfem::assembler
{
	namespace
	{
		inline RowVectorNd zero_point(const int dim)
		{
			assert(dim == 2 || dim == 3);
			return RowVectorNd::Zero(dim);
		}
	} // namespace

	Helmholtz::Helmholtz()
		: k_("k")
	{
	}

	template <int element_dim>
	void Helmholtz::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		assert(local_element_matrix.size() == 1);
		double res = 0;
		for (int q = 0; q < data.quad_num; ++q)
		{
			double x, y, z;
			data.fill_phy_position<element_dim>(q, x, y, z);
			double k = k_(x, y, z, data.time, data.element_id);

			auto gradi = data.gather_basis_grad_phy<element_dim>(data.row_local_basis_id, q);
			auto gradj = data.gather_basis_grad_phy<element_dim>(data.col_local_basis_id, q);
			res += gradi.dot(gradj) * data.weighted_measure[q];
			res -= data.gather_basis_value(data.row_local_basis_id, q)
				   * data.gather_basis_value(data.col_local_basis_id, q)
				   * data.weighted_measure[q] * k * k;
		}

		local_element_matrix[0] = res;
	}

	void Helmholtz::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
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

	VectorNd Helmholtz::compute_rhs(const AutodiffHessianPt &pt) const
	{
		Eigen::Matrix<double, 1, 1> result;
		assert(pt.size() == 1);
		const RowVectorNd val = zero_point(pt(0).getHessian().rows());

		const double tmp = k_(val, 0, 0);
		result(0) = pt(0).getHessian().trace() + tmp * tmp * pt(0).getValue();
		return result;
	}

	void Helmholtz::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		k_.add_multimaterial(index, params, "", root_path);
	}

	Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> Helmholtz::kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const
	{
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(1);

		const RowVectorNd val = zero_point(dim);
		const double tmp = k_(val, 0, 0);

		if (dim == 2)
			res(0) = -0.25 * utils::bessy0(tmp * r);
		else if (dim == 3)
			res(0) = 0.25 * cos(tmp * r) / (M_PI * r);
		else
			assert(false);

		return res;
	}

	std::map<std::string, Assembler::ParamFunc> Helmholtz::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		res["k"] = [this](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return this->k_(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler
