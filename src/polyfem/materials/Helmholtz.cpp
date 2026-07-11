#include <polyfem/materials/Helmholtz.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	HelmholtzExpr HelmholtzExpr::from_json(const json &j, const Units &, const std::string &root_path)
	{
		HelmholtzExpr out;
		out.k = parse_expr(j.at("k"), "", root_path);
		return out;
	}

	Helmholtz HelmholtzExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Helmholtz out{};
		out.k = k(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
