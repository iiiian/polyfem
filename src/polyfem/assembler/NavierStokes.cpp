#include "NavierStokes.hpp"

#include <algorithm>

namespace polyfem::assembler
{

	NavierStokesVelocity::NavierStokesVelocity()
		: viscosity_("viscosity")
	{
	}

	void NavierStokesVelocity::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		viscosity_.add_multimaterial(index, params, units.viscosity(), root_path);
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	NavierStokesVelocity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res(size());

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> val(size());
		for (int d = 0; d < size(); ++d)
			val(d) = pt(d).getValue();

		const auto nu = viscosity_(val, 0, 0);
		for (int d = 0; d < size(); ++d)
		{
			res(d) = -val.dot(pt(d).getGradient()) + nu * pt(d).getHessian().trace();
		}

		return res;
	}

	void NavierStokesVelocity::assemble_gradient(const NonLinearElementAssemblyData &data, span<double> local_gradient) const
	{
		assert(false);
		std::fill(local_gradient.begin(), local_gradient.end(), 0.0);
	}

	void NavierStokesVelocity::assemble_hessian(const NonLinearElementAssemblyData &data, span<double> local_hessian) const
	{
		Eigen::MatrixXd H;
		if (full_gradient_)
			H = compute_N(data) + compute_W(data);
		else
			H = compute_N(data);

		int local_matrix_size = data.basis_num * size();
		assert(local_hessian.size() == local_matrix_size * local_matrix_size);
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> local_hessian_mat(
			local_hessian.data(), local_matrix_size, local_matrix_size);
		local_hessian_mat = H.transpose();
	}

	// Compute N = int v \cdot \nabla phi_i \cdot \phi_j

	Eigen::MatrixXd NavierStokesVelocity::compute_N(const NonLinearElementAssemblyData &data) const
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> GradMat;

		assert(data.x.cols() == 1);

		const int n_pts = data.weighted_measure.size();
		const int n_bases = data.basis_num;

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * size(), 1);
		local_vel.setZero();
		for (int i = 0; i < n_bases; ++i)
		{
			auto ids = data.global_ids(i);
			auto weights = data.global_weights(i);
			for (int ii = 0; ii < ids.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_vel(i * size() + d) += weights[ii] * data.x(ids[ii] * size() + d, 0);
				}
			}
		}

		Eigen::MatrixXd N(size() * n_bases, size() * n_bases);
		N.setZero();

		GradMat grad_i(size(), size());
		GradMat jac_it(size(), size());

		Eigen::VectorXd vel(size(), 1);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> phi_j(size(), 1);

		for (long p = 0; p < n_pts; ++p)
		{
			vel.setZero();

			for (int i = 0; i < n_bases; ++i)
			{
				const double val = data.gather_basis_value(i, p);

				for (int d = 0; d < size(); ++d)
				{
					vel(d) += val * local_vel(i * size() + d);
				}
			}

			jac_it = data.jacobian_inverse_transpose(p);

			for (int i = 0; i < n_bases; ++i)
			{
				for (int m = 0; m < size(); ++m)
				{
					grad_i.setZero();
					if (size() == 2)
						grad_i.row(m) = data.gather_basis_grad<2>(i, p);
					else
						grad_i.row(m) = data.gather_basis_grad<3>(i, p);
					grad_i = grad_i * jac_it;

					for (int j = 0; j < n_bases; ++j)
					{
						for (int n = 0; n < size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = data.gather_basis_value(j, p);
							N(i * size() + m, j * size() + n) += (grad_i * vel).dot(phi_j) * data.weighted_measure[p];
						}
					}
				}
			}
		}

		return N;
	}

	// Compute N = int phi_j \cdot \nabla v \cdot \phi_j

	Eigen::MatrixXd NavierStokesVelocity::compute_W(const NonLinearElementAssemblyData &data) const
	{
		typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> GradMat;

		assert(data.x.cols() == 1);

		const int n_pts = data.weighted_measure.size();
		const int n_bases = data.basis_num;

		Eigen::Matrix<double, Eigen::Dynamic, 1> local_vel(n_bases * size(), 1);
		local_vel.setZero();
		for (int i = 0; i < n_bases; ++i)
		{
			auto ids = data.global_ids(i);
			auto weights = data.global_weights(i);
			for (int ii = 0; ii < ids.size(); ++ii)
			{
				for (int d = 0; d < size(); ++d)
				{
					local_vel(i * size() + d) += weights[ii] * data.x(ids[ii] * size() + d, 0);
				}
			}
		}

		Eigen::MatrixXd W(size() * n_bases, size() * n_bases);
		W.setZero();

		GradMat grad_v(size(), size());
		GradMat jac_it(size(), size());

		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> phi_i(size(), 1);
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> phi_j(size(), 1);

		for (long p = 0; p < n_pts; ++p)
		{
			grad_v.setZero();

			for (int i = 0; i < n_bases; ++i)
			{
				RowVectorNd grad;
				if (size() == 2)
					grad = data.gather_basis_grad<2>(i, p);
				else
					grad = data.gather_basis_grad<3>(i, p);
				assert(grad.size() == size());

				for (int d = 0; d < size(); ++d)
				{
					for (int c = 0; c < size(); ++c)
					{
						grad_v(d, c) += grad(c) * local_vel(i * size() + d);
					}
				}
			}

			jac_it = data.jacobian_inverse_transpose(p);
			grad_v = (grad_v * jac_it).eval();

			for (int i = 0; i < n_bases; ++i)
			{
				for (int m = 0; m < size(); ++m)
				{
					phi_i.setZero();
					phi_i(m) = data.gather_basis_value(i, p);

					for (int j = 0; j < n_bases; ++j)
					{
						for (int n = 0; n < size(); ++n)
						{
							phi_j.setZero();
							phi_j(n) = data.gather_basis_value(j, p);
							W(i * size() + m, j * size() + n) += (grad_v * phi_i).dot(phi_j) * data.weighted_measure[p];
						}
					}
				}
			}
		}

		return W;
	}

	std::map<std::string, Assembler::ParamFunc> NavierStokesVelocity::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &nu = viscosity_;
		res["viscosity"] = [&nu](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return nu(p, t, e);
		};

		return res;
		return res;
	}
} // namespace polyfem::assembler
