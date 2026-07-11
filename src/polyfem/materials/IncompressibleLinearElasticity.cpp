#include <polyfem/materials/IncompressibleLinearElasticity.hpp>
#include <polyfem/materials/MaterialUtils.hpp>

namespace polyfem::material
{
	IncompressibleLinearElasticityExpr IncompressibleLinearElasticityExpr::from_json(const json &j, const Units &units, const std::string &root_path)
	{
		IncompressibleLinearElasticityExpr out;
		out.lame = LameParameter<utils::ExpressionValue>::from_json(j, units, root_path);
		return out;
	}

	IncompressibleLinearElasticity IncompressibleLinearElasticityExpr::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		IncompressibleLinearElasticity out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
