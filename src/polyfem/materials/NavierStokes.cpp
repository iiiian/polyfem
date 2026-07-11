#include <polyfem/materials/NavierStokes.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	NavierStokesExpr NavierStokesExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		NavierStokesExpr out;
		out.viscosity = parse_expr(j.at("viscosity"), units.viscosity(), root_path);
		return out;
	}

	NavierStokes NavierStokesExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		NavierStokes out{};
		out.viscosity = viscosity(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
