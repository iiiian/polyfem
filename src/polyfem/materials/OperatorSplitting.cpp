#include <polyfem/materials/OperatorSplitting.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	OperatorSplittingExpr OperatorSplittingExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		OperatorSplittingExpr out;
		out.viscosity = parse_expr(j.at("viscosity"), units.viscosity(), root_path);
		return out;
	}

	OperatorSplitting OperatorSplittingExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		OperatorSplitting out{};
		out.viscosity = viscosity(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
