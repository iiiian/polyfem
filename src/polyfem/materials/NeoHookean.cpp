#include <polyfem/materials/NeoHookean.hpp>

namespace polyfem::material
{
	NeoHookean<utils::ExpressionValue> neo_hookean_from_json(const json &j, const Units &units, const std::string &root_path)
	{
		NeoHookean<utils::ExpressionValue> out;
		out.lame = lame_parameter_from_json(j, units, root_path);
		return out;
	}

	NeoHookean<double> eval_expr(const NeoHookean<utils::ExpressionValue> &expr, double x, double y, double z, double t, int element_id)
	{
		NeoHookean<double> out{};
		out.lame = eval_expr(expr.lame, x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
