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
	};

	LameParameter<utils::ExpressionValue> lame_parameter_from_json(
		const json &j,
		const Units &units,
		const std::string &root_path);

	LameParameter<double> eval_expr(
		const LameParameter<utils::ExpressionValue> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1);

#ifdef POLYFEM_WITH_CUDA
	inline LameParameter<utils::ExpressionValueView> make_device_expr(
		const LameParameter<utils::ExpressionValue> &expr,
		ExecutionPolicy policy)
	{
		using Host = LameParameter<utils::ExpressionValue>;
		using View = LameParameter<utils::ExpressionValueView>;
		const auto type = expr.type == Host::Type::YoungPoisson
							  ? View::Type::YoungPoisson
							  : View::Type::LambdaMu;
		return {
			type,
			expr.E.device_view(policy),
			expr.nu.device_view(policy),
			expr.lambda.device_view(policy),
			expr.mu.device_view(policy)};
	}

	inline bool is_device_compatible(
		const LameParameter<utils::ExpressionValue> &expr)
	{
		if (expr.type == LameParameter<utils::ExpressionValue>::Type::YoungPoisson)
			return expr.E.is_device_compatible() && expr.nu.is_device_compatible();
		return expr.lambda.is_device_compatible() && expr.mu.is_device_compatible();
	}

	POLYFEM_BOTH inline LameParameter<double> eval_expr(
		const LameParameter<utils::ExpressionValueView> &expr,
		double x,
		double y,
		double z = 0,
		double t = 0,
		int element_id = -1)
	{
		LameParameter<double> out{};
		if (expr.type == LameParameter<utils::ExpressionValueView>::Type::YoungPoisson)
		{
			out.type = LameParameter<double>::Type::YoungPoisson;
			out.E = expr.E(x, y, z, t, element_id);
			out.nu = expr.nu(x, y, z, t, element_id);
		}
		else
		{
			out.type = LameParameter<double>::Type::LambdaMu;
			out.lambda = expr.lambda(x, y, z, t, element_id);
			out.mu = expr.mu(x, y, z, t, element_id);
		}
		return out;
	}
#endif

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
