#pragma once

#include <polyfem/assembler/GenericElastic.hpp>
#include <polyfem/assembler/MatParams.hpp>

/// Used for test only
namespace polyfem::assembler
{
	class SimplifiedNeoHookeanAutodiff : public GenericElastic<SimplifiedNeoHookeanAutodiff>
	{
	public:
		std::string name() const override { return "SimplifiedNeoHookeanAudodiff"; }
		// Used only for testing, no need to output
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

		SimplifiedNeoHookeanAutodiff();

		// sets material params
		void add_multimaterial(const int index, const json &params, const Units &units) override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			double lambda, mu;
			params_.lambda_mu(p, p, t, el_id, lambda, mu);

			const T t_lambda{lambda};
			const T t_mu{mu};

			// simplified neohookean parameter shift
			const T snh_lambda = t_lambda + 5.0f / 6.0f * t_mu;
			const T snh_mu = 4.0f / 3.0f * t_mu;

			const T alpha = 1.0f + 3.0f / 4.0f * snh_mu / snh_lambda;
			const T Ic = (def_grad * def_grad.transpose()).trace();
			const T J = polyfem::utils::determinant(def_grad);
			const T J_delta = J - alpha;
			const T val = snh_mu / 2 * (Ic - 3.0f) + snh_lambda / 2.0f * J_delta * J_delta - snh_mu / 2.0f * log(Ic + 1.0f);

			return val;
		}

	private:
		LameParameters params_;
	};
} // namespace polyfem::assembler
