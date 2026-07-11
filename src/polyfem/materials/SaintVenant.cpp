#include <polyfem/materials/SaintVenant.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	template <int dim>
	SaintVenantExpr<dim> SaintVenantExpr<dim>::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		SaintVenantExpr<dim> out;
		if (j.contains("elasticity_tensor"))
		{
			out.elasticity_tensor = ElasticityTensor<utils::ExpressionValue, dim>::from_json(j.at("elasticity_tensor"), units, root_path);
		}
		else
		{
			out.lame = LameParameter<utils::ExpressionValue>::from_json(j, units, root_path);
		}

		out.fiber_direction = j.contains("fiber_direction")
								  ? FiberDirection<utils::ExpressionValue, dim>::from_json(j.at("fiber_direction"), units, root_path)
								  : FiberDirection<utils::ExpressionValue, dim>::from_json(json::array(), units, root_path);
		return out;
	}

	template <int dim>
	SaintVenant<dim> SaintVenantExpr<dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		SaintVenant<dim> out{};
		out.fiber_direction = fiber_direction.eval_expr(x, y, z, t, element_id);

		if (elasticity_tensor.depends_on_elastic_coeff)
		{
			out.lame = lame.eval_expr(x, y, z, t, element_id);
			out.elasticity_tensor = ElasticityTensor<double, dim>::from_lame(out.lame);
		}
		else
		{
			out.elasticity_tensor = elasticity_tensor.eval_expr(x, y, z, t, element_id);
		}

		maybe_rotate_elasticity_tensor(out.fiber_direction, out.elasticity_tensor);
		return out;
	}
	template struct SaintVenantExpr<1>;
	template struct SaintVenantExpr<2>;
	template struct SaintVenantExpr<3>;
} // namespace polyfem::material
