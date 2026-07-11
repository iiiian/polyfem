#include <polyfem/materials/MooneyRivlin.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	MooneyRivlinExpr MooneyRivlinExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		const std::string stress = units.stress();
		MooneyRivlinExpr out;
		out.c1 = parse_expr(j.at("c1"), stress, root_path);
		out.c2 = parse_expr(j.at("c2"), stress, root_path);
		out.k = parse_expr(j.at("k"), stress, root_path);
		return out;
	}

	MooneyRivlin MooneyRivlinExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		MooneyRivlin out{};
		out.c1 = c1(x, y, z, t, element_id);
		out.c2 = c2(x, y, z, t, element_id);
		out.k = k(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
