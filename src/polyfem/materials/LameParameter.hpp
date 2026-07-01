#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <string>
#include <utility>

namespace polyfem::material
{
	template <typename Scalar>
	struct LameParameter
	{
		enum class Type
		{
			YoungPoisson,
			LambdaMu
		};

		Type type = Type::LambdaMu;
		Scalar E;
		Scalar nu;
		Scalar lambda;
		Scalar mu;

		static LameParameter<utils::ExpressionValue> from_json(const json &j, const Units &units, const std::string &root_path);

		/// Evaluate lame expression at position (x,y,z), time t, and element id.
		LameParameter<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	/// Get (lambda, mu) from LameParameter. Convert from E/nu if necessary.
	template <int dim>
	POLYFEM_BOTH std::pair<double, double> lambda_mu(const LameParameter<double> &lame)
	{
		if (lame.type == LameParameter<double>::Type::LambdaMu)
		{
			return {lame.lambda, lame.mu};
		}

		double E = lame.E;
		double nu = lame.nu;
		double mu = E / (2.0 * (1.0 + nu));
		if constexpr (dim == 1)
		{
			return {0.0, E / 2.0};
		}
		else if constexpr (dim == 2)
		{
			return {(nu * E) / (1.0 - nu * nu), mu};
		}
		else
		{
			return {(E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)), mu};
		}
	}
} // namespace polyfem::material
