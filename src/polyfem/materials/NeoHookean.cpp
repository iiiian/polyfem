#include <polyfem/materials/NeoHookean.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	NeoHookeanExpr NeoHookeanExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		NeoHookeanExpr out;
		out.lame = LameParameter<utils::ExpressionValue>::from_json(j, units, root_path);
		return out;
	}

	NeoHookean NeoHookeanExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		NeoHookean out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
