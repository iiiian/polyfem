#include <polyfem/materials/FixedCorotational.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	FixedCorotationalExpr FixedCorotationalExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		FixedCorotationalExpr out;
		out.lame = LameParameter<utils::ExpressionValue>::from_json(j, units, root_path);
		return out;
	}

	FixedCorotational FixedCorotationalExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		FixedCorotational out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
