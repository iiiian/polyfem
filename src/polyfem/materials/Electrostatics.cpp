#include <polyfem/materials/Electrostatics.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	ElectrostaticsExpr ElectrostaticsExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		ElectrostaticsExpr out;
		out.epsilon = parse_expr(j.at("epsilon"), units.permittivity(), root_path);
		return out;
	}

	Electrostatics ElectrostaticsExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Electrostatics out{};
		out.epsilon = epsilon(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
