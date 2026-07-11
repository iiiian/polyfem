#include <polyfem/materials/MooneyRivlin3ParamSymbolic.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	MooneyRivlin3ParamSymbolicExpr MooneyRivlin3ParamSymbolicExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		const std::string stress = units.stress();
		MooneyRivlin3ParamSymbolicExpr out;
		out.c1 = parse_expr(j.at("c1"), stress, root_path);
		out.c2 = parse_expr(j.at("c2"), stress, root_path);
		out.c3 = parse_expr(j.at("c3"), stress, root_path);
		out.d1 = parse_expr(j.at("d1"), stress, root_path);
		return out;
	}

	MooneyRivlin3ParamSymbolic MooneyRivlin3ParamSymbolicExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		MooneyRivlin3ParamSymbolic out{};
		out.c1 = c1(x, y, z, t, element_id);
		out.c2 = c2(x, y, z, t, element_id);
		out.c3 = c3(x, y, z, t, element_id);
		out.d1 = d1(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
