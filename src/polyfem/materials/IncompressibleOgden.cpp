#include <polyfem/materials/IncompressibleOgden.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	IncompressibleOgdenExpr IncompressibleOgdenExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		const std::string stress = units.stress();
		IncompressibleOgdenExpr out;
		out.c = parse_ogden_vector(j.at("c"), stress, root_path);
		out.m = parse_ogden_vector(j.at("m"), "", root_path);
		out.k = parse_expr(j.at("k"), stress, root_path);
		return out;
	}

	IncompressibleOgden IncompressibleOgdenExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		IncompressibleOgden out{};
		out.c = eval_ogden_vector(c, x, y, z, t, element_id);
		out.m = eval_ogden_vector(m, x, y, z, t, element_id);
		out.k = k(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
