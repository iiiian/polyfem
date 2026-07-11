#include <polyfem/materials/UnconstrainedOgden.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	UnconstrainedOgdenExpr UnconstrainedOgdenExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		const std::string stress = units.stress();
		UnconstrainedOgdenExpr out;
		out.alphas = parse_ogden_vector(j.at("alphas"), "", root_path);
		out.mus = parse_ogden_vector(j.at("mus"), stress, root_path);
		out.Ds = parse_ogden_vector(j.at("Ds"), stress, root_path);
		return out;
	}

	UnconstrainedOgden UnconstrainedOgdenExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		UnconstrainedOgden out{};
		out.alphas = eval_ogden_vector(alphas, x, y, z, t, element_id);
		out.mus = eval_ogden_vector(mus, x, y, z, t, element_id);
		out.Ds = eval_ogden_vector(Ds, x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
