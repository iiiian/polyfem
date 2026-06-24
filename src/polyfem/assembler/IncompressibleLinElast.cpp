#include "IncompressibleLinElast.hpp"

namespace polyfem::assembler
{
	using namespace basis;

	void IncompressibleLinearElasticityDispacement::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() >= 1 && size() <= 3);

		params_.add_multimaterial(index, params, size() == 3, units.stress(), root_path);
	}

	template <int element_dim>
	void IncompressibleLinearElasticityDispacement::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		// 2mu (epsi : epsj)
		assert(size() == element_dim);
		assert(local_element_matrix.size() == size() * size());

		Eigen::Matrix<double, element_dim, element_dim> epsi;
		Eigen::Matrix<double, element_dim, element_dim> epsj;

		for (int q = 0; q < data.quad_num; ++q)
		{
			double lambda, mu;
			double x, y, z;
			data.fill_phy_position<element_dim>(q, x, y, z);
			params_.lambda_mu(0.0, 0.0, 0.0, x, y, z, data.time, data.element_id, lambda, mu);

			auto gradi = data.gather_basis_grad_phy<element_dim>(data.row_local_basis_id, q);
			auto gradj = data.gather_basis_grad_phy<element_dim>(data.col_local_basis_id, q);
			for (int di = 0; di < size(); ++di)
			{
				epsi.setZero();
				epsi.row(di) = gradi.transpose();
				epsi = ((epsi + epsi.transpose()) / 2).eval();
				for (int dj = 0; dj < size(); ++dj)
				{
					epsj.setZero();
					epsj.row(dj) = gradj.transpose();
					epsj = ((epsj + epsj.transpose()) / 2.0).eval();

					local_element_matrix[di * size() + dj] += 2 * mu * (epsi.array() * epsj.array()).sum() * data.weighted_measure[q];
				}
			}
		}
	}

	void IncompressibleLinearElasticityDispacement::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
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

	void IncompressibleLinearElasticityDispacement::assign_stress_tensor(const OutputData &data,
																		 const int all_size,
																		 const ElasticityTensorType &type,
																		 Eigen::MatrixXd &all,
																		 const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		const auto &displacement = data.fun;
		const auto &local_pts = data.local_pts;
		const auto &bs = data.bs;
		const auto &gbs = data.gbs;
		const auto el_id = data.el_id;

		assert(size() == 2 || size() == 3);
		all.resize(local_pts.rows(), all_size);
		assert(displacement.cols() == 1);

		Eigen::MatrixXd displacement_grad(size(), size());

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			if (type == ElasticityTensorType::F)
			{
				all.row(p) = fun(displacement_grad + Eigen::MatrixXd::Identity(size(), size()));
				continue;
			}

			double lambda, mu;
			params_.lambda_mu(local_pts.row(p), vals.val.row(p), data.t, vals.element_id, lambda, mu);

			const Eigen::MatrixXd strain = (displacement_grad + displacement_grad.transpose()) / 2;
			Eigen::MatrixXd stress = 2 * mu * strain + lambda * strain.trace() * Eigen::MatrixXd::Identity(size(), size());

			if (type == ElasticityTensorType::PK1)
				stress = pk1_from_cauchy(stress, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));
			else if (type == ElasticityTensorType::PK2)
				stress = pk2_from_cauchy(stress, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));

			all.row(p) = fun(stress);
		}
	}

	std::map<std::string, Assembler::ParamFunc> IncompressibleLinearElasticityDispacement::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &params = params_;
		const int size = this->size();

		res["lambda"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);
			return lambda;
		};

		res["mu"] = [&params](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);
			return mu;
		};

		res["E"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;
			params.lambda_mu(uv, p, t, e, lambda, mu);

			if (size == 3)
				return mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu);
			else
				return 2 * mu * (2.0 * lambda + 2.0 * mu) / (lambda + 2.0 * mu);
		};

		res["nu"] = [&params, size](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
			double lambda, mu;

			params.lambda_mu(uv, p, t, e, lambda, mu);

			if (size == 3)
				return lambda / (2.0 * (lambda + mu));
			else
				return lambda / (lambda + 2.0 * mu);
		};

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	IncompressibleLinearElasticityMixed::assemble(const MixedAssemblerData &data) const
	{
		// (psii : div phij)  = -psii * gradphij
		assert(size() == 2 || size() == 3);
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

	void IncompressibleLinearElasticityPressure::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(disp_size_ >= 1 && disp_size_ <= 3);

		params_.add_multimaterial(index, params, disp_size_ == 3, units.stress(), root_path);
	}

	template <int element_dim>
	void IncompressibleLinearElasticityPressure::assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
	{
		// -1/lambda phi_ * phi_j
		assert(local_element_matrix.size() == 1);

		double res = 0;

		for (int q = 0; q < data.quad_num; ++q)
		{
			double lambda, mu;
			double x, y, z;
			data.fill_phy_position<element_dim>(q, x, y, z);
			params_.lambda_mu(0.0, 0.0, 0.0, x, y, z, data.time, data.element_id, lambda, mu);

			res += -data.gather_basis_value(data.row_local_basis_id, q)
				   * data.gather_basis_value(data.col_local_basis_id, q)
				   * data.weighted_measure[q] / lambda;
		}

		local_element_matrix[0] = res;
	}

	void IncompressibleLinearElasticityPressure::assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const
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
