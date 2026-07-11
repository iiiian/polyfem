#include <polyfem/materials/Stokes.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	StokesExpr StokesExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		StokesExpr out;
		out.viscosity = parse_expr(j.at("viscosity"), units.viscosity(), root_path);
		return out;
	}

	Stokes StokesExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Stokes out{};
		out.viscosity = viscosity(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
