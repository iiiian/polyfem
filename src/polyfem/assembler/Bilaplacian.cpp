#include "Bilaplacian.hpp"

namespace polyfem::assembler
{
	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	BilaplacianMixed::assemble(const MixedAssemblerData &data) const
	{
		const Eigen::MatrixXd &gradi = data.psi_vals.basis_values[data.i].grad_t_m;
		const Eigen::MatrixXd &gradj = data.phi_vals.basis_values[data.j].grad_t_m;

		// return ((psii.array() * phij.array()).rowwise().sum().array() * da.array()).colwise().sum();
		double res = 0;
		for (int k = 0; k < gradi.rows(); ++k)
		{
			res += gradi.row(k).dot(gradj.row(k)) * data.da(k);
		}
		return Eigen::Matrix<double, 1, 1>::Constant(res);
	}

	template <int element_dim>
	void BilaplacianAux::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		assert(local_element_matrix.size() == 1);
		double tmp = 0;
		for (int q = 0; q < data.quad_num; ++q)
		{
			tmp += data.gather_basis_value(data.row_local_basis_id, q)
				   * data.gather_basis_value(data.col_local_basis_id, q)
				   * data.weighted_measure[q];
		}
		local_element_matrix[0] = tmp;
	}

	void BilaplacianAux::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
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
