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
	struct MooneyRivlin3ParamSymbolicExpr;

	struct MooneyRivlin3ParamSymbolic
	{
		using ExprType = MooneyRivlin3ParamSymbolicExpr;

		double c1;
		double c2;
		double c3;
		double d1;
	};

	struct MooneyRivlin3ParamSymbolicExpr
	{

		utils::ExpressionValue c1;
		utils::ExpressionValue c2;
		utils::ExpressionValue c3;
		utils::ExpressionValue d1;

		static MooneyRivlin3ParamSymbolicExpr from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		MooneyRivlin3ParamSymbolic eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
