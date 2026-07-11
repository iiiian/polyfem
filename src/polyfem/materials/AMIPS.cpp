#include <polyfem/materials/AMIPS.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	AMIPSExpr AMIPSExpr::from_json(const json &j, const Units &, const std::string &root_path)
	{
		AMIPSExpr out;
		out.use_rest_pose = j.at("use_rest_pose").get<bool>();
		out.weight = parse_expr(j.at("weight"), "", root_path);
		return out;
	}

	AMIPS AMIPSExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		AMIPS out{};
		out.use_rest_pose = use_rest_pose;
		out.weight = weight(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
