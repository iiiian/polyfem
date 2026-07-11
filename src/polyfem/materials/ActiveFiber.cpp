#include <polyfem/materials/ActiveFiber.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	template <int dim>
	ActiveFiberExpr<dim> ActiveFiberExpr<dim>::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		const std::string stress = units.stress();
		ActiveFiberExpr<dim> out;
		out.activation = parse_expr(j.at("activation"), "", root_path);
		out.Tmax = parse_expr(j.at("Tmax"), stress, root_path);
		out.fiber_direction = j.contains("fiber_direction")
								  ? FiberDirection<utils::ExpressionValue, dim>::from_json(j.at("fiber_direction"), units, root_path)
								  : FiberDirection<utils::ExpressionValue, dim>::from_json(json::array(), units, root_path);
		return out;
	}

	template <int dim>
	ActiveFiber<dim> ActiveFiberExpr<dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		ActiveFiber<dim> out{};
		out.activation = activation(x, y, z, t, element_id);
		out.Tmax = Tmax(x, y, z, t, element_id);
		out.fiber_direction = fiber_direction.eval_expr(x, y, z, t, element_id);
		return out;
	}
	template struct ActiveFiberExpr<1>;
	template struct ActiveFiberExpr<2>;
	template struct ActiveFiberExpr<3>;
} // namespace polyfem::material
