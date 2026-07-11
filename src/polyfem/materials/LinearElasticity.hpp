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
	struct LinearElasticityExpr;

	struct LinearElasticity
	{
		using ExprType = LinearElasticityExpr;

		LameParameter<double> lame;
	};

	struct LinearElasticityExpr
	{

		LameParameter<utils::ExpressionValue> lame;

		static LinearElasticityExpr from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		LinearElasticity eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
