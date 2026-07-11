#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/materials/LameParameter.hpp>
#include <polyfem/materials/ElasticityTensor.hpp>
#include <polyfem/materials/FiberDirection.hpp>

#include <Eigen/Core>

#include <string>

namespace polyfem::material
{
	struct MooneyRivlin3ParamExpr;

	struct MooneyRivlin3Param
	{
		using ExprType = MooneyRivlin3ParamExpr;

		double c1;
		double c2;
		double c3;
		double d1;
	};

	struct MooneyRivlin3ParamExpr
	{

		utils::ExpressionValue c1;
		utils::ExpressionValue c2;
		utils::ExpressionValue c3;
		utils::ExpressionValue d1;

		static MooneyRivlin3ParamExpr from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		MooneyRivlin3Param eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
