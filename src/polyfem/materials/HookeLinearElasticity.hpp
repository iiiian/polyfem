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
	template <int dim>
	struct HookeLinearElasticityExpr;

	template <int dim>
	struct HookeLinearElasticity
	{
		using ExprType = HookeLinearElasticityExpr<dim>;

		LameParameter<double> lame;
		ElasticityTensor<double, dim> elasticity_tensor;
		FiberDirection<double, dim> fiber_direction;
	};

	template <int dim>
	struct HookeLinearElasticityExpr
	{

		LameParameter<utils::ExpressionValue> lame;
		ElasticityTensor<utils::ExpressionValue, dim> elasticity_tensor;
		FiberDirection<utils::ExpressionValue, dim> fiber_direction;

		static HookeLinearElasticityExpr from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		HookeLinearElasticity<dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
