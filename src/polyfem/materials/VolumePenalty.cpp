#include <polyfem/materials/VolumePenalty.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	VolumePenaltyExpr VolumePenaltyExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		VolumePenaltyExpr out;
		out.k = parse_expr(j.at("k"), units.stress(), root_path);
		return out;
	}

	VolumePenalty VolumePenaltyExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		VolumePenalty out{};
		out.k = k(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
