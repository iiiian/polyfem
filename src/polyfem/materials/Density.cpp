#include <polyfem/materials/Density.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	DensityExpr DensityExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		DensityExpr out;
		if (j.contains("rho"))
		{
			out.rho = parse_expr(j.at("rho"), units.density(), root_path);
		}
		else
		{
			out.rho = parse_expr(j.at("density"), units.density(), root_path);
		}
		return out;
	}

	Density DensityExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Density out{};
		out.rho = rho(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
