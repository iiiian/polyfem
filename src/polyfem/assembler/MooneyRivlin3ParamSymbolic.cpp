#include "MooneyRivlin3ParamSymbolic.hpp"

#include <polyfem/autogen/auto_mooney_rivlin_gradient_hessian.hpp>

namespace polyfem::assembler
{

	MooneyRivlin3ParamSymbolic::MooneyRivlin3ParamSymbolic()
		: c1_("c1"), c2_("c2"), c3_("c3"), d1_("d1")
	{
	}

	void MooneyRivlin3ParamSymbolic::assemble_gradient(const NonLinearElementAssemblyData &data, span<double> local_gradient) const
	{
		assert(local_gradient.size() == data.basis_num * size());
		if (size() == 2)
		{
			switch (data.basis_num)
			{
			case 3:
			{
				compute_energy_aux_gradient_fast<3, 2>(data, local_gradient);
				break;
			}
			case 6:
			{
				compute_energy_aux_gradient_fast<6, 2>(data, local_gradient);
				break;
			}
			case 10:
			{
				compute_energy_aux_gradient_fast<10, 2>(data, local_gradient);
				break;
			}
			default:
			{
				compute_energy_aux_gradient_fast<Eigen::Dynamic, 2>(data, local_gradient);
				break;
			}
			}
		}
		else // if (size() == 3)
		{
			assert(size() == 3);
			switch (data.basis_num)
			{
			case 4:
			{
				compute_energy_aux_gradient_fast<4, 3>(data, local_gradient);
				break;
			}
			case 10:
			{
				compute_energy_aux_gradient_fast<10, 3>(data, local_gradient);
				break;
			}
			case 20:
			{
				compute_energy_aux_gradient_fast<20, 3>(data, local_gradient);
				break;
			}
			default:
			{
				compute_energy_aux_gradient_fast<Eigen::Dynamic, 3>(data, local_gradient);
				break;
			}
			}
		}
	}

	void MooneyRivlin3ParamSymbolic::assemble_hessian(const NonLinearElementAssemblyData &data, span<double> local_hessian) const
	{
		int local_matrix_size = data.basis_num * size();
		assert(local_hessian.size() == local_matrix_size * local_matrix_size);
		std::fill(local_hessian.begin(), local_hessian.end(), 0.0);

		if (size() == 2)
		{
			switch (data.basis_num)
			{
			case 3:
			{
				compute_energy_hessian_aux_fast<3, 2>(data, local_hessian);
				break;
			}
			case 6:
			{
				compute_energy_hessian_aux_fast<6, 2>(data, local_hessian);
				break;
			}
			case 10:
			{
				compute_energy_hessian_aux_fast<10, 2>(data, local_hessian);
				break;
			}
			default:
			{
				compute_energy_hessian_aux_fast<Eigen::Dynamic, 2>(data, local_hessian);
				break;
			}
			}
		}
		else // if (size() == 3)
		{
			assert(size() == 3);
			switch (data.basis_num)
			{
			case 4:
			{
				compute_energy_hessian_aux_fast<4, 3>(data, local_hessian);
				break;
			}
			case 10:
			{
				compute_energy_hessian_aux_fast<10, 3>(data, local_hessian);
				break;
			}
			case 20:
			{
				compute_energy_hessian_aux_fast<20, 3>(data, local_hessian);
				break;
			}
			default:
			{
				compute_energy_hessian_aux_fast<Eigen::Dynamic, 3>(data, local_hessian);
				break;
			}
			}
		}
	}

	void MooneyRivlin3ParamSymbolic::assign_stress_tensor(const OutputData &data,
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
		const auto I = Eigen::MatrixXd::Identity(size(), size());

		for (long p = 0; p < local_pts.rows(); ++p)
		{
			compute_diplacement_grad(size(), bs, vals, local_pts, p, displacement, displacement_grad);

			const Eigen::MatrixXd def_grad = I + displacement_grad;
			const Eigen::MatrixXd def_grad_T = def_grad.transpose();
			if (type == ElasticityTensorType::F)
			{
				all.row(p) = fun(def_grad);
				continue;
			}

			const double t = 0;
			const double c1 = c1_(local_pts.row(p), t, vals.element_id);
			const double c2 = c2_(local_pts.row(p), t, vals.element_id);
			const double c3 = c3_(local_pts.row(p), t, vals.element_id);
			const double d1 = d1_(local_pts.row(p), t, vals.element_id);

			Eigen::MatrixXd stress_tensor;
			autogen::generate_gradient(c1, c2, c3, d1, def_grad_T, stress_tensor);

			stress_tensor = 1.0 / def_grad.determinant() * stress_tensor * def_grad.transpose();
			if (type == ElasticityTensorType::PK1)
				stress_tensor = pk1_from_cauchy(stress_tensor, def_grad);
			else if (type == ElasticityTensorType::PK2)
				stress_tensor = pk2_from_cauchy(stress_tensor, def_grad);

			all.row(p) = fun(stress_tensor);
		}
	}

	void MooneyRivlin3ParamSymbolic::compute_stress_grad_multiply_mat(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &mat,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < size(); ++d)
			F(d, d) += 1.;
		Eigen::MatrixXd F_T = F.transpose();

		const double c1 = c1_(global_pts, t, el_id);
		const double c2 = c2_(global_pts, t, el_id);
		const double c3 = c3_(global_pts, t, el_id);
		const double d1 = d1_(global_pts, t, el_id);

		Eigen::MatrixXd grad, hess;
		get_grad_hess_symbolic(c1, c2, c3, d1, F, grad, hess);
		// get_grad_hess_autodiff(c1, c2, c3, d1, global_pts, el_id, F, grad, hess);

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		// Compute ∂S_ij/∂F_kl * M_kl, same as M_ij * ∂S_ij/∂F_kl since the hessian is symmetric
		result = (hess * mat.reshaped(size() * size(), 1)).reshaped(size(), size());
	}

	void MooneyRivlin3ParamSymbolic::compute_stress_grad_multiply_stress(
		const OptAssemblerData &data,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{
		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < size(); ++d)
			F(d, d) += 1.;

		const double c1 = c1_(global_pts, t, el_id);
		const double c2 = c2_(global_pts, t, el_id);
		const double c3 = c3_(global_pts, t, el_id);
		const double d1 = d1_(global_pts, t, el_id);

		Eigen::MatrixXd grad, hess;
		get_grad_hess_symbolic(c1, c2, c3, d1, F, grad, hess);
		// get_grad_hess_autodiff(c1, c2, c3, d1, global_pts, el_id, F, grad, hess);

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		// Compute ∂S_ij/∂F_kl * S_kl, same as S_ij * ∂S_ij/∂F_kl since the hessian is symmetric
		result = (hess * stress.reshaped(size() * size(), 1)).reshaped(size(), size());
	}

	void MooneyRivlin3ParamSymbolic::compute_stress_grad_multiply_vect(
		const OptAssemblerData &data,
		const Eigen::MatrixXd &vect,
		Eigen::MatrixXd &stress,
		Eigen::MatrixXd &result) const
	{

		const double t = data.t;
		const int el_id = data.el_id;
		const Eigen::MatrixXd &local_pts = data.local_pts;
		const Eigen::MatrixXd &global_pts = data.global_pts;
		const Eigen::MatrixXd &grad_u_i = data.grad_u_i;

		Eigen::MatrixXd F = grad_u_i;
		for (int d = 0; d < size(); ++d)
			F(d, d) += 1.;
		Eigen::MatrixXd F_T = F.transpose();

		const double c1 = c1_(global_pts, t, el_id);
		const double c2 = c2_(global_pts, t, el_id);
		const double c3 = c3_(global_pts, t, el_id);
		const double d1 = d1_(global_pts, t, el_id);

		Eigen::MatrixXd grad, hess;
		get_grad_hess_symbolic(c1, c2, c3, d1, F, grad, hess);
		// get_grad_hess_autodiff(c1, c2, c3, d1, global_pts, el_id, F, grad, hess);

		// Stress is S_ij = ∂W(F)/∂F_ij
		stress = grad;
		result.resize(hess.rows(), vect.size());
		for (int i = 0; i < hess.rows(); ++i)
			if (vect.rows() == 1)
				// Compute ∂S_ij/∂F_kl * v_k, same as ∂S_ij/∂F_kl * v_i since the hessian is symmetric
				result.row(i) = vect * hess.row(i).reshaped(size(), size());
			else
				// Compute ∂S_ij/∂F_kl * v_l, same as ∂S_ij/∂F_kl * v_j since the hessian is symmetric
				result.row(i) = hess.row(i).reshaped(size(), size()) * vect;
	}

	template <typename T>
	T MooneyRivlin3ParamSymbolic::elastic_energy(
		const RowVectorNd &p,
		const int el_id,
		const AutoDiffGradMat<T> &def_grad) const
	{
		const double t = 0; // TODO

		const double c1 = c1_(p, t, el_id);
		const double c2 = c2_(p, t, el_id);
		const double c3 = c3_(p, t, el_id);
		const double d1 = d1_(p, t, el_id);

		const T J = polyfem::utils::determinant(def_grad);
		const T log_J = log(J);
		const auto F_tilde = def_grad;
		const auto right_cauchy_green = F_tilde * F_tilde.transpose();

		const auto I1_tilde = pow(J, -2. / size()) * first_invariant(right_cauchy_green);
		const auto I2_tilde = pow(J, -4. / size()) * second_invariant(right_cauchy_green);

		const T val = c1 * (I1_tilde - size()) + c2 * (I2_tilde - size()) + c3 * (I1_tilde - size()) * (I2_tilde - size()) + d1 * log_J * log_J;

		return val;
	}

	void MooneyRivlin3ParamSymbolic::get_grad_hess_symbolic(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &F, Eigen::MatrixXd &grad, Eigen::MatrixXd &hess) const
	{
		Eigen::MatrixXd F_T = F.transpose();

		// Grad is ∂W(F)/∂F_ij
		autogen::generate_gradient(c1, c2, c3, d1, F_T, grad);
		// Hessian is ∂W(F)/(∂F_ij*∂F_kl)
		autogen::generate_hessian(c1, c2, c3, d1, F, hess);
	}

	void MooneyRivlin3ParamSymbolic::get_grad_hess_autodiff(const double c1, const double c2, const double c3, const double d1, const Eigen::MatrixXd &global_pts, const int el_id, const Eigen::MatrixXd &F, Eigen::MatrixXd &grad, Eigen::MatrixXd &hess) const
	{
		typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>> Diff;

		DiffScalarBase::setVariableCount(size() * size());

		Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> def_grad(size(), size());

		for (int i = 0; i < size(); ++i)
			for (int j = 0; j < size(); ++j)
				def_grad(i, j) = Diff(i + j * size(), F(i, j));

		const auto energy = elastic_energy(global_pts, el_id, def_grad);

		// Grad is ∂W(F)/∂F_ij
		grad = energy.getGradient().reshaped(size(), size());
		// Hessian is ∂W(F)/(∂F_ij*∂F_kl)
		hess = energy.getHessian();
	}

	double MooneyRivlin3ParamSymbolic::compute_energy(const NonLinearElementAssemblyData &data) const
	{
		return compute_energy_aux<double>(data);
	}

	template <typename T>
	T MooneyRivlin3ParamSymbolic::compute_energy_aux(const NonLinearElementAssemblyData &data) const
	{
		AutoDiffVect<T> local_disp;
		get_local_disp(data, size(), local_disp);

		AutoDiffGradMat<T> def_grad(size(), size());

		T energy = T(0.0);

		const int n_pts = data.weighted_measure.size();
		for (long p = 0; p < n_pts; ++p)
		{
			compute_disp_grad_at_quad(data, local_disp, p, size(), def_grad);

			// Id + grad d
			for (int d = 0; d < size(); ++d)
				def_grad(d, d) += T(1);

			const T val = elastic_energy(data.physical_position(p), data.element_id, def_grad);

			energy += val * data.weighted_measure[p];
		}
		return energy;
	}

	template <int n_basis, int dim>
	void MooneyRivlin3ParamSymbolic::compute_energy_aux_gradient_fast(const NonLinearElementAssemblyData &data, span<double> local_gradient) const
	{
		assert(data.x.cols() == 1);
		Eigen::Map<Eigen::VectorXd> G_flattened(local_gradient.data(), local_gradient.size());

		const int n_pts = data.weighted_measure.size();

		Eigen::Matrix<double, n_basis, dim> local_disp(data.basis_num, size());
		data.gather_unknowns<dim>(local_disp);

		Eigen::Matrix<double, dim, dim> def_grad(size(), size());
		Eigen::Matrix<double, dim, dim> def_grad_T(size(), size());

		Eigen::Matrix<double, n_basis, dim> G(data.basis_num, size());
		G.setZero();

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.basis_num, size());
			data.gather_basis_grads<dim>(p, grad);

			Eigen::Matrix<double, dim, dim> jac_it = data.gather_J_inverse_transpose<dim>(p);
			Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;

			// Id + grad d
			def_grad = local_disp.transpose() * delF_delU + Eigen::Matrix<double, dim, dim>::Identity(size(), size());
			def_grad_T = def_grad.transpose();

			const double t = 0;
			const double c1 = c1_(data.physical_position(p), t, data.element_id);
			const double c2 = c2_(data.physical_position(p), t, data.element_id);
			const double c3 = c3_(data.physical_position(p), t, data.element_id);
			const double d1 = d1_(data.physical_position(p), t, data.element_id);

			Eigen::Matrix<double, dim, dim> gradient_temp;
			autogen::generate_gradient_templated<dim>(c1, c2, c3, d1, def_grad_T, gradient_temp);

			Eigen::Matrix<double, n_basis, dim> gradient = delF_delU * gradient_temp.transpose();
			G.noalias() += gradient * data.weighted_measure[p];
		}

		Eigen::Matrix<double, dim, n_basis> G_T = G.transpose();

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		Eigen::Matrix<double, N, 1> temp(Eigen::Map<Eigen::Matrix<double, N, 1>>(G_T.data(), G_T.size()));
		G_flattened = temp;
	}

	template <int n_basis, int dim>
	void MooneyRivlin3ParamSymbolic::compute_energy_hessian_aux_fast(const NonLinearElementAssemblyData &data, span<double> local_hessian) const
	{
		assert(data.x.cols() == 1);

		constexpr int N = (n_basis == Eigen::Dynamic) ? Eigen::Dynamic : n_basis * dim;
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> H(local_hessian.data(), data.basis_num * dim, data.basis_num * dim);
		const int n_pts = data.weighted_measure.size();

		Eigen::Matrix<double, n_basis, dim> local_disp(data.basis_num, size());
		data.gather_unknowns<dim>(local_disp);

		Eigen::Matrix<double, dim, dim> def_grad(size(), size());

		for (long p = 0; p < n_pts; ++p)
		{
			Eigen::Matrix<double, n_basis, dim> grad(data.basis_num, size());
			data.gather_basis_grads<dim>(p, grad);

			Eigen::Matrix<double, dim, dim> jac_it = data.gather_J_inverse_transpose<dim>(p);
			const Eigen::Matrix<double, n_basis, dim> delF_delU = grad * jac_it;

			// Id + grad d
			def_grad = local_disp.transpose() * delF_delU + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

			const double t = 0;
			const double c1 = c1_(data.physical_position(p), t, data.element_id);
			const double c2 = c2_(data.physical_position(p), t, data.element_id);
			const double c3 = c3_(data.physical_position(p), t, data.element_id);
			const double d1 = d1_(data.physical_position(p), t, data.element_id);

			Eigen::Matrix<double, dim * dim, dim * dim> hessian_temp;
			autogen::generate_hessian_templated<dim>(c1, c2, c3, d1, def_grad, hessian_temp);

			Eigen::Matrix<double, dim * dim, N> delF_delU_tensor = Eigen::Matrix<double, dim * dim, N>::Zero(jac_it.size(), grad.size());

			for (size_t i = 0; i < local_disp.rows(); ++i)
			{
				for (size_t j = 0; j < local_disp.cols(); ++j)
				{
					Eigen::Matrix<double, dim, dim> temp(size(), size());
					temp.setZero();
					temp.row(j) = delF_delU.row(i);
					Eigen::Matrix<double, dim * dim, 1> temp_flattened(Eigen::Map<Eigen::Matrix<double, dim * dim, 1>>(temp.data(), temp.size()));
					delF_delU_tensor.col(i * size() + j) = temp_flattened;
				}
			}

			Eigen::Matrix<double, N, N> hessian = delF_delU_tensor.transpose() * hessian_temp * delF_delU_tensor;

			H += hessian * data.weighted_measure[p];
		}
	}

	void MooneyRivlin3ParamSymbolic::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		c1_.add_multimaterial(index, params, units.stress(), root_path);
		c2_.add_multimaterial(index, params, units.stress(), root_path);
		c3_.add_multimaterial(index, params, units.stress(), root_path);
		d1_.add_multimaterial(index, params, units.stress(), root_path);
	}

	std::map<std::string, Assembler::ParamFunc> MooneyRivlin3ParamSymbolic::parameters() const
	{
		std::map<std::string, ParamFunc> res;
		const auto &c1 = this->c1();
		const auto &c2 = this->c2();
		const auto &c3 = this->c3();
		const auto &d1 = this->d1();

		res["c1"] = [&c1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c1(p, t, e);
		};

		res["c2"] = [&c2](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c2(p, t, e);
		};

		res["c3"] = [&c3](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return c3(p, t, e);
		};

		res["d1"] = [&d1](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return d1(p, t, e);
		};

		return res;
	}

} // namespace polyfem::assembler
