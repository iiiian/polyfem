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
	struct HGOFiberExpr;

	template <int dim>
	struct HGOFiber
	{
		using ExprType = HGOFiberExpr<dim>;

		double k1;
		double k2;
		FiberDirection<double, dim> fiber_direction;
	};

	template <int dim>
	struct HGOFiberExpr
	{

		utils::ExpressionValue k1;
		utils::ExpressionValue k2;
		FiberDirection<utils::ExpressionValue, dim> fiber_direction;

		static HGOFiberExpr from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		HGOFiber<dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
