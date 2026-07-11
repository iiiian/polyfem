#include <polyfem/materials/IsochoricNeoHookean.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	IsochoricNeoHookeanExpr IsochoricNeoHookeanExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		IsochoricNeoHookeanExpr out;
		out.lame = LameParameter<utils::ExpressionValue>::from_json(j, units, root_path);
		return out;
	}

	IsochoricNeoHookean IsochoricNeoHookeanExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		IsochoricNeoHookean out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
