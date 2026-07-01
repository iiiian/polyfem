#include <polyfem/materials/LameParameter.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/CudaBoth.hpp>

#include <string>
#include <stdexcept>

namespace polyfem::material
{
	namespace
	{
		using Expr = utils::ExpressionValue;

		Expr parse_expr(const json &value, const std::string &unit_type, const std::string &root_path)
		{
			Expr out;
			out.init(value, root_path);
			out.set_unit_type(unit_type);
			return out;
		}
	} // namespace

	template <typename Scalar>
	LameParameter<Expr> LameParameter<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		bool has_E = j.contains("E");
		bool has_young = j.contains("young");
		bool has_nu = j.contains("nu");
		bool has_lambda = j.contains("lambda");
		bool has_mu = j.contains("mu");

		bool has_young_poisson = has_E || has_young || has_nu;
		bool has_lambda_mu = has_lambda || has_mu;

		const std::string stress = units.stress();
		LameParameter<Expr> out;
		if (has_young_poisson)
		{
			out.type = LameParameter<Expr>::Type::YoungPoisson;
			out.E = parse_expr(j.at(has_young ? "young" : "E"), stress, root_path);
			out.nu = parse_expr(j.at("nu"), "", root_path);
			return out;
		}

		if (has_lambda_mu)
		{
			out.type = LameParameter<Expr>::Type::LambdaMu;
			out.lambda = parse_expr(j.at("lambda"), stress, root_path);
			out.mu = parse_expr(j.at("mu"), stress, root_path);
			return out;
		}

		throw std::runtime_error("LameParameter: expected young/nu, E/nu, or lambda/mu");
	}

	template LameParameter<Expr> LameParameter<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	LameParameter<double> LameParameter<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		LameParameter<double> out{};
		if (type == LameParameter<Scalar>::Type::YoungPoisson)
		{
			out.type = LameParameter<double>::Type::YoungPoisson;
			out.E = E(x, y, z, t, element_id);
			out.nu = nu(x, y, z, t, element_id);
		}
		else
		{
			out.type = LameParameter<double>::Type::LambdaMu;
			out.lambda = lambda(x, y, z, t, element_id);
			out.mu = mu(x, y, z, t, element_id);
		}
		return out;
	}

	template LameParameter<double> LameParameter<Expr>::eval_expr(double, double, double, double, int) const;
} // namespace polyfem::material
