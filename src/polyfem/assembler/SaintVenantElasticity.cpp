#include "SaintVenantElasticity.hpp"

#include <polyfem/autogen/auto_elasticity_rhs.hpp>

#include <algorithm>

namespace polyfem::assembler
{
	namespace
	{
		template <class Matrix>
		Matrix strain_from_disp_grad(const Matrix &disp_grad)
		{
			// Matrix mat =  (disp_grad + disp_grad.transpose());
			Matrix mat = (disp_grad.transpose() * disp_grad + disp_grad + disp_grad.transpose());

			for (int i = 0; i < mat.size(); ++i)
				mat(i) *= 0.5;

			return mat;
		}
	} // namespace

	SaintVenantElasticity::SaintVenantElasticity()
	{
	}

	void SaintVenantElasticity::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		if (!params.contains("elasticity_tensor"))
		{
			if (params.count("young"))
			{
				if (params["young"].is_number() && params["nu"].is_number())
					elasticity_tensor_.set_from_young_poisson(params["young"], params["nu"], units.stress(), root_path);
			}
			else if (params.count("E"))
			{
				if (params["E"].is_number() && params["nu"].is_number())
					elasticity_tensor_.set_from_young_poisson(params["E"], params["nu"], units.stress(), root_path);
			}
			else if (params.count("lambda"))
			{
				if (params["lambda"].is_number() && params["mu"].is_number())
					elasticity_tensor_.set_from_lambda_mu(params["lambda"], params["mu"], units.stress(), root_path);
			}
		}
		else
		{
			std::vector<double> entries = params["elasticity_tensor"];
			elasticity_tensor_.set_from_entries(entries, units.stress(), root_path);
		}
	}

	void SaintVenantElasticity::set_size(const int size)
	{
		Assembler::set_size(size);
		elasticity_tensor_.resize(size);
	}

	template <typename T, unsigned long N>
	T SaintVenantElasticity::stress(const ElasticityTensor &elasticity_tensor, const std::array<T, N> &strain, const int j) const
	{
		T res = elasticity_tensor(j, 0) * strain[0];

		for (unsigned long k = 1; k < N; ++k)
			res += elasticity_tensor(j, k) * strain[k];

		return res;
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	SaintVenantElasticity::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;

		if (size() == 2)
			autogen::saint_venant_2d_function(pt, elasticity_tensor_, res);
		else if (size() == 3)
			autogen::saint_venant_3d_function(pt, elasticity_tensor_, res);
		else
			assert(false);

		return res;
	}

	void SaintVenantElasticity::assemble_gradient(const NonLinearElementAssemblyData &data, span<double> local_gradient) const
	{
		// igl::Timer time; time.start();

		const int n_bases = data.basis_num;

		Eigen::VectorXd gradient = polyfem::gradient_from_energy(
			size(), n_bases, data,
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 6, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 8, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 12, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 18, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 24, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 30, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 60, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, 81, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, BIG_N, 1>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar1<double, Eigen::VectorXd>>(data); });
		assert(gradient.size() == local_gradient.size());
		std::copy(gradient.data(), gradient.data() + gradient.size(), local_gradient.data());
	}

	void SaintVenantElasticity::assemble_hessian(const NonLinearElementAssemblyData &data, span<double> local_hessian) const
	{
		// igl::Timer time; time.start();

		const int n_bases = data.basis_num;
		Eigen::MatrixXd hessian = polyfem::hessian_from_energy(
			size(), n_bases, data,
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 8, 1>, Eigen::Matrix<double, 8, 8>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 18, 1>, Eigen::Matrix<double, 18, 18>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 24, 1>, Eigen::Matrix<double, 24, 24>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 30, 1>, Eigen::Matrix<double, 30, 30>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 60, 1>, Eigen::Matrix<double, 60, 60>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, 81, 1>, Eigen::Matrix<double, 81, 81>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, SMALL_N, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, SMALL_N, SMALL_N>>>(data); },
			[&](const NonLinearElementAssemblyData &data) { return compute_energy_aux<DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd>>(data); });
		int local_matrix_size = n_bases * size();
		assert(local_hessian.size() == local_matrix_size * local_matrix_size);
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> local_hessian_mat(
			local_hessian.data(), local_matrix_size, local_matrix_size);
		local_hessian_mat = hessian;
	}

	void SaintVenantElasticity::assign_stress_tensor(
		const OutputData &data,
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
		const auto t = data.t;

		Eigen::MatrixXd displacement_grad(size(), size());

		assert(displacement.cols() == 1);

		all.resize(local_pts.rows(), all_size);

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

			Eigen::MatrixXd strain = strain_from_disp_grad(displacement_grad);
			Eigen::MatrixXd stress_tensor(size(), size());

			if (size() == 2)
			{
				std::array<double, 3> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = 2 * strain(0, 1);

				stress_tensor << stress(elasticity_tensor_, eps, 0), stress(elasticity_tensor_, eps, 2),
					stress(elasticity_tensor_, eps, 2), stress(elasticity_tensor_, eps, 1);
			}
			else
			{
				std::array<double, 6> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = strain(2, 2);
				eps[3] = 2 * strain(1, 2);
				eps[4] = 2 * strain(0, 2);
				eps[5] = 2 * strain(0, 1);

				stress_tensor << stress(elasticity_tensor_, eps, 0), stress(elasticity_tensor_, eps, 5), stress(elasticity_tensor_, eps, 4),
					stress(elasticity_tensor_, eps, 5), stress(elasticity_tensor_, eps, 1), stress(elasticity_tensor_, eps, 3),
					stress(elasticity_tensor_, eps, 4), stress(elasticity_tensor_, eps, 3), stress(elasticity_tensor_, eps, 2);
			}

			stress_tensor = (Eigen::MatrixXd::Identity(size(), size()) + displacement_grad) * stress_tensor;

			if (type == ElasticityTensorType::PK1)
				stress_tensor = pk1_from_cauchy(stress_tensor, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));
			else if (type == ElasticityTensorType::PK2)
				stress_tensor = pk2_from_cauchy(stress_tensor, displacement_grad + Eigen::MatrixXd::Identity(size(), size()));

			all.row(p) = fun(stress_tensor);
		}
	}

	double SaintVenantElasticity::compute_energy(const NonLinearElementAssemblyData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	// Compute \int \sigma : E
	template <typename T>
	T SaintVenantElasticity::compute_energy_aux(const NonLinearElementAssemblyData &data) const
	{
		typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
		typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;

		AutoDiffVect local_disp;
		get_local_disp(data, size(), local_disp);

		AutoDiffGradMat disp_grad(size(), size());

		T energy = T(0.0);

		const int n_pts = data.weighted_measure.size();
		for (long p = 0; p < n_pts; ++p)
		{
			compute_disp_grad_at_quad(data, local_disp, p, size(), disp_grad);

			// AutoDiffGradMat fdir_mtx_voigt_ad(fdir_mtx_voigt.rows(), fdir_mtx_voigt.cols());
			// for (int i = 0; i < fdir_mtx_voigt.rows(); ++i)
			// {
			// 	for (int j = 0; j < fdir_mtx_voigt.cols(); ++j)
			// 	{
			// 		fdir_mtx_voigt_ad(i, j) = T(fdir_mtx_voigt(i, j));
			// 	}
			// }

			AutoDiffGradMat strain = strain_from_disp_grad(disp_grad);
			AutoDiffGradMat stress_tensor(size(), size());

			if (size() == 2)
			{
				std::array<T, 3> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = 2 * strain(0, 1);

				stress_tensor << stress(elasticity_tensor_, eps, 0), stress(elasticity_tensor_, eps, 2),
					stress(elasticity_tensor_, eps, 2), stress(elasticity_tensor_, eps, 1);
			}
			else
			{
				std::array<T, 6> eps;
				eps[0] = strain(0, 0);
				eps[1] = strain(1, 1);
				eps[2] = strain(2, 2);
				eps[3] = 2 * strain(1, 2);
				eps[4] = 2 * strain(0, 2);
				eps[5] = 2 * strain(0, 1);

				stress_tensor << stress(elasticity_tensor_, eps, 0), stress(elasticity_tensor_, eps, 5), stress(elasticity_tensor_, eps, 4),
					stress(elasticity_tensor_, eps, 5), stress(elasticity_tensor_, eps, 1), stress(elasticity_tensor_, eps, 3),
					stress(elasticity_tensor_, eps, 4), stress(elasticity_tensor_, eps, 3), stress(elasticity_tensor_, eps, 2);
			}

			energy += (stress_tensor * strain).trace() * data.weighted_measure[p];
		}

		return energy * 0.5;
	}

	std::map<std::string, Assembler::ParamFunc> SaintVenantElasticity::parameters() const
	{
		std::map<std::string, ParamFunc> res;

		const auto &elast_tensor = this->elasticity_tensor_;
		const int size = this->size() == 2 ? 3 : 6;

		for (int i = 0; i < size; ++i)
		{
			for (int j = i; j < size; ++j)
			{
				res[fmt::format("C_{}{}", i, j)] = [&elast_tensor, i, j](const RowVectorNd &, const RowVectorNd &, double, int) {
					return elast_tensor(i, j);
				};
			}
		}
		return res;
	}
} // namespace polyfem::assembler
