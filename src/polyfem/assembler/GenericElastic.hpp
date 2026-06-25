#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/ElasticityUtils.hpp>

#include <polyfem/utils/Logger.hpp>

// non linear NeoHookean material model
namespace polyfem::assembler
{
	enum class AutodiffType
	{
		FULL,
		STRESS,
		NONE
	};

	template <typename T>
	using DefGradMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>;

	template <typename Derived>
	class GenericElastic : public ElasticityNLAssembler
	{
	public:
		using ElasticityNLAssembler::assemble_energy;
		using ElasticityNLAssembler::assemble_gradient;
		using ElasticityNLAssembler::assemble_hessian;

		GenericElastic();
		virtual ~GenericElastic() = default;

		// energy, gradient, and hessian used in newton method
		double compute_energy(const NonLinearElementAssemblyData &data) const override;
		void assemble_hessian(const NonLinearElementAssemblyData &data, span<double> local_hessian) const override;
		void assemble_gradient(const NonLinearElementAssemblyData &data, span<double> local_gradient) const override;

		void assign_stress_tensor(const OutputData &data,
								  const int all_size,
								  const ElasticityTensorType &type,
								  Eigen::MatrixXd &all,
								  const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const override;

		void compute_stress_grad_multiply_mat(const OptAssemblerData &data,
											  const Eigen::MatrixXd &mat,
											  Eigen::MatrixXd &stress,
											  Eigen::MatrixXd &result) const override;

		void compute_stress_grad_multiply_stress(const OptAssemblerData &data,
												 Eigen::MatrixXd &stress,
												 Eigen::MatrixXd &result) const override;

		void compute_stress_grad_multiply_vect(const OptAssemblerData &data,
											   const Eigen::MatrixXd &vect,
											   Eigen::MatrixXd &stress,
											   Eigen::MatrixXd &result) const override;

		/// @brief Returns this as a reference to derived class
		Derived &derived() { return static_cast<Derived &>(*this); }
		/// @brief Returns this as a const reference to derived class
		const Derived &derived() const { return static_cast<const Derived &>(*this); }

		// sets material params
		virtual void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override = 0;

		bool allow_inversion() const override { return true; }

		virtual bool real_def_grad() const { return true; }

	protected:
		AutodiffType autodiff_type_ = AutodiffType::STRESS;

		virtual Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> gradient(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
		{
			log_and_throw_error("gradient not implemented");
		}

		virtual Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hessian(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
		{
			log_and_throw_error("hessian not implemented");
		}

	private:
		// utility function that computes energy, the template is used for double, DScalar1, and DScalar2 in energy, gradient and hessian
		template <typename T>
		T compute_energy_aux(const NonLinearElementAssemblyData &data) const
		{
			typedef Eigen::Matrix<T, Eigen::Dynamic, 1> AutoDiffVect;
			typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;
			typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> DoubleGradMat;

			AutoDiffVect local_disp;
			get_local_disp(data, size(), local_disp);

			AutoDiffGradMat def_grad(size(), size());

			T energy = T(0.0);

			const int n_pts = data.weighted_measure.size();
			for (long p = 0; p < n_pts; ++p)
			{
				compute_disp_grad_at_quad(data, local_disp, p, size(), def_grad);

				// Id + grad d
				for (int d = 0; d < size(); ++d)
					def_grad(d, d) += T(1);

				if (!derived().real_def_grad())
				{
					DoubleGradMat tmp_jac_it = data.jacobian_inverse_transpose(p);
					tmp_jac_it = tmp_jac_it.inverse();

					AutoDiffGradMat jac_it(size(), size());
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = T(tmp_jac_it(k));

					def_grad *= jac_it;
				}

				const T val = derived().elastic_energy(data.physical_position(p), data.t, data.element_id, def_grad);

				energy += val * data.weighted_measure[p];
			}
			return energy;
		}

		void assemble_hessian_full_ad(const NonLinearElementAssemblyData &data, span<double> local_hessian) const;
		void assemble_gradient_full_ad(const NonLinearElementAssemblyData &data, span<double> local_gradient) const;

		void assemble_hessian_stress_ad(const NonLinearElementAssemblyData &data, span<double> local_hessian) const;
		void assemble_gradient_stress_ad(const NonLinearElementAssemblyData &data, span<double> local_gradient) const;

		void assemble_hessian_stress_noad(const NonLinearElementAssemblyData &data, span<double> local_hessian) const;
		void assemble_gradient_stress_noad(const NonLinearElementAssemblyData &data, span<double> local_gradient) const;

		template <int n_basis, int dim>
		void compute_gradient_from_stress(const NonLinearElementAssemblyData &data, Eigen::VectorXd &res) const
		{
			typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>> Diff;
			typedef Eigen::Matrix<Diff, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;
			typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> DoubleGradMat;

			const int n_pts = data.weighted_measure.size();

			Eigen::Matrix<double, n_basis, dim> local_disp(data.basis_num, size());
			data.gather_unknowns<dim>(local_disp);

			Eigen::Matrix<double, dim, dim> def_grad(size(), size());
			DiffScalarBase::setVariableCount(size() * size());
			AutoDiffGradMat def_grad_ad(size(), size());

			res.resize(data.basis_num * size());
			res.setZero();

			for (long p = 0; p < n_pts; ++p)
			{
				Eigen::Matrix<double, n_basis, dim> grad(data.basis_num, size());
				data.gather_basis_grads<dim>(p, grad);

				Eigen::Matrix<double, dim, dim> jac_it = data.gather_J_inverse_transpose<dim>(p);
				Eigen::Matrix<double, n_basis, dim> G = grad * jac_it;
				def_grad = local_disp.transpose() * G + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

				for (int d1 = 0; d1 < size(); ++d1)
					for (int d2 = 0; d2 < size(); ++d2)
						def_grad_ad(d1, d2) = Diff(d1 * size() + d2, def_grad(d1, d2));

				if (!derived().real_def_grad())
				{
					DoubleGradMat tmp_jac_it = data.gather_J_inverse_transpose<dim>(p);
					tmp_jac_it = tmp_jac_it.inverse();

					AutoDiffGradMat jac_it(size(), size());
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = Diff(tmp_jac_it(k));

					def_grad_ad *= jac_it;
				}

				const Diff val = derived().elastic_energy(data.physical_position(p), data.t, data.element_id, def_grad_ad);

				Eigen::Matrix<double, dim, dim> P;
				for (int i = 0; i < size(); ++i)
					for (int j = 0; j < size(); ++j)
						P(i, j) = val.getGradient()(i * size() + j);

				const Eigen::Matrix<double, n_basis, dim> Rloc = G * P.transpose() * data.weighted_measure[p];

				for (int a = 0; a < data.basis_num; ++a)
					for (int d = 0; d < size(); ++d)
						res(a * size() + d) += Rloc(a, d);
			}
		}

		template <int n_basis, int dim>
		void compute_gradient_from_stress_noad(const NonLinearElementAssemblyData &data, Eigen::VectorXd &res) const
		{
			typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> DoubleGradMat;

			const int n_pts = data.weighted_measure.size();

			Eigen::Matrix<double, n_basis, dim> local_disp(data.basis_num, size());
			data.gather_unknowns<dim>(local_disp);

			Eigen::Matrix<double, dim, dim> def_grad(size(), size());
			res.resize(data.basis_num * size());
			res.setZero();

			for (long p = 0; p < n_pts; ++p)
			{
				Eigen::Matrix<double, n_basis, dim> grad(data.basis_num, size());
				data.gather_basis_grads<dim>(p, grad);

				Eigen::Matrix<double, dim, dim> jac_it = data.gather_J_inverse_transpose<dim>(p);
				Eigen::Matrix<double, n_basis, dim> G = grad * jac_it;
				def_grad = local_disp.transpose() * G + Eigen::Matrix<double, dim, dim>::Identity(size(), size());

				if (!derived().real_def_grad())
				{
					DoubleGradMat jac_it = data.gather_J_inverse_transpose<dim>(p);
					jac_it = jac_it.inverse();

					def_grad *= jac_it;
				}

				const Eigen::Matrix<double, dim, dim> P = derived().gradient(data.physical_position(p), data.t, data.element_id, def_grad);
				const Eigen::Matrix<double, n_basis, dim> Bgrad = derived().real_def_grad() ? G : grad;
				const Eigen::Matrix<double, n_basis, dim> Rloc = Bgrad * P.transpose() * data.weighted_measure[p];

				for (int a = 0; a < data.basis_num; ++a)
					for (int d = 0; d < size(); ++d)
						res(a * size() + d) += Rloc(a, d);
			}
		}

		template <int n_basis, int dim>
		void compute_hessian_from_stress(const NonLinearElementAssemblyData &data, Eigen::MatrixXd &H) const
		{
			typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9>> Diff2;
			typedef Eigen::Matrix<Diff2, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AutoDiffGradMat;
			typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> DoubleGradMat;

			const int d = size();
			const int nb = data.basis_num;
			const int n_pts = data.weighted_measure.size();

			Eigen::Matrix<double, n_basis, dim> local_disp(nb, d);
			data.gather_unknowns<dim>(local_disp);

			DiffScalarBase::setVariableCount(d * d);
			AutoDiffGradMat def_grad_ad(d, d);
			Eigen::Matrix<double, dim, dim> def_grad(d, d);

			H.resize(nb * d, nb * d);
			H.setZero();

			for (long p = 0; p < n_pts; ++p)
			{
				Eigen::Matrix<double, n_basis, dim> grad_ref(nb, d);
				data.gather_basis_grads<dim>(p, grad_ref);

				const Eigen::Matrix<double, dim, dim> jac_it = data.gather_J_inverse_transpose<dim>(p);
				const Eigen::Matrix<double, n_basis, dim> G = grad_ref * jac_it;

				def_grad = local_disp.transpose() * G + Eigen::Matrix<double, dim, dim>::Identity();

				for (int i = 0; i < d; ++i)
					for (int j = 0; j < d; ++j)
						def_grad_ad(i, j) = Diff2(i * d + j, def_grad(i, j));

				if (!derived().real_def_grad())
				{
					DoubleGradMat tmp_jac_it = data.gather_J_inverse_transpose<dim>(p);
					tmp_jac_it = tmp_jac_it.inverse();

					AutoDiffGradMat jac_it(size(), size());
					for (long k = 0; k < jac_it.size(); ++k)
						jac_it(k) = Diff2(tmp_jac_it(k));

					def_grad_ad *= jac_it;
				}

				const Diff2 val = derived().elastic_energy(data.physical_position(p), data.t, data.element_id, def_grad_ad);

				Eigen::Matrix<double, dim * dim, dim * dim> A(d * d, d * d);
				A.setZero();

				// Material tangent A = d vec(P) / d vec(F)
				// Since P = dW/dF, A is just the Hessian of W wrt vec(F).
				for (int r = 0; r < d * d; ++r)
					for (int c = 0; c < d * d; ++c)
						A(r, c) = val.getHessian()(r, c);

				for (int a = 0; a < nb; ++a)
				{
					const Eigen::Matrix<double, dim * dim, dim> Ba = compute_B_block<dim>(G.row(a));

					for (int b = 0; b < nb; ++b)
					{
						const Eigen::Matrix<double, dim * dim, dim> Bb = compute_B_block<dim>(G.row(b));

						const Eigen::Matrix<double, dim, dim> Kab =
							Ba.transpose() * A * Bb * data.weighted_measure[p];

						H.template block<dim, dim>(a * d, b * d) += Kab;
					}
				}
			}
		}

		template <int n_basis, int dim>
		void compute_hessian_from_stress_noad(const NonLinearElementAssemblyData &data, Eigen::MatrixXd &H) const
		{
			typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> DoubleGradMat;

			const int d = size();
			const int nb = data.basis_num;
			const int n_pts = data.weighted_measure.size();

			Eigen::Matrix<double, n_basis, dim> local_disp(nb, d);
			data.gather_unknowns<dim>(local_disp);

			Eigen::Matrix<double, dim, dim> def_grad(d, d);

			H.resize(nb * d, nb * d);
			H.setZero();

			for (long p = 0; p < n_pts; ++p)
			{
				Eigen::Matrix<double, n_basis, dim> grad_ref(nb, d);
				data.gather_basis_grads<dim>(p, grad_ref);

				const Eigen::Matrix<double, dim, dim> jac_it = data.gather_J_inverse_transpose<dim>(p);
				const Eigen::Matrix<double, n_basis, dim> G = grad_ref * jac_it;

				def_grad = local_disp.transpose() * G + Eigen::Matrix<double, dim, dim>::Identity();

				if (!derived().real_def_grad())
				{
					DoubleGradMat jac_it = data.gather_J_inverse_transpose<dim>(p);
					jac_it = jac_it.inverse();

					def_grad *= jac_it;
				}

				// Material tangent A = d vec(P) / d vec(F)
				// Since P = dW/dF, A is just the Hessian of W wrt vec(F).
				Eigen::Matrix<double, dim * dim, dim * dim> A = derived().hessian(data.physical_position(p), data.t, data.element_id, def_grad);

				const Eigen::Matrix<double, n_basis, dim> Bgrad = derived().real_def_grad() ? G : grad_ref;

				for (int a = 0; a < nb; ++a)
				{
					const Eigen::Matrix<double, dim * dim, dim> Ba = compute_B_block<dim>(Bgrad.row(a));

					for (int b = 0; b < nb; ++b)
					{
						const Eigen::Matrix<double, dim * dim, dim> Bb = compute_B_block<dim>(Bgrad.row(b));

						const Eigen::Matrix<double, dim, dim> Kab =
							Ba.transpose() * A * Bb * data.weighted_measure[p];

						H.template block<dim, dim>(a * d, b * d) += Kab;
					}
				}
			}
		}

		template <int dim>
		Eigen::Matrix<double, dim * dim, dim> compute_B_block(const Eigen::Matrix<double, 1, dim> &g) const;
	};
} // namespace polyfem::assembler
