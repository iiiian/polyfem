#include "Laplacian.hpp"

namespace polyfem::assembler
{
	namespace
	{
		bool delta(int i, int j)
		{
			return (i == j) ? true : false;
		}

		template <int dim>
		void assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix)
		{
			assert(local_element_matrix.size() == 1);
			double res = 0;
			for (int q = 0; q < data.quad_num; ++q)
			{
				auto gradi = data.gather_basis_grad_phy<dim>(data.row_local_basis_id, q);
				auto gradj = data.gather_basis_grad_phy<dim>(data.col_local_basis_id, q);
				res += gradi.dot(gradj) * data.weighted_measure[q];
			}
			local_element_matrix[0] = res;
		}
	} // namespace

	void Laplacian::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
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

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> Laplacian::compute_rhs(const AutodiffHessianPt &pt) const
	{
		Eigen::Matrix<double, 1, 1> result;
		assert(pt.size() == 1);
		result(0) = pt(0).getHessian().trace();
		return result;
	}

	Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> Laplacian::kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const
	{
		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> res(1);

		if (dim == 2)
			res(0) = -1. / (2 * M_PI) * log(r);
		else if (dim == 3)
			res(0) = 1. / (4 * M_PI * r);
		else
			assert(false);

		return res;
	}

	void Laplacian::compute_stress_grad_multiply_mat(const OptAssemblerData &data,
													 const Eigen::MatrixXd &mat,
													 Eigen::MatrixXd &stress,
													 Eigen::MatrixXd &result) const
	{
		stress = data.grad_u_i;
		result = mat;
	}

	void Laplacian::compute_stiffness_value(const double t,
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
} // namespace polyfem::assembler
