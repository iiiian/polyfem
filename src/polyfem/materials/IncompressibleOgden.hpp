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
	struct IncompressibleOgdenExpr;

	struct IncompressibleOgden
	{
		using ExprType = IncompressibleOgdenExpr;

		// Max order 6.
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 6, 1> c;
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 6, 1> m;
		double k;
	};

	struct IncompressibleOgdenExpr
	{

		// Max order 6.
		Eigen::Matrix<utils::ExpressionValue, Eigen::Dynamic, 1, 0, 6, 1> c;
		Eigen::Matrix<utils::ExpressionValue, Eigen::Dynamic, 1, 0, 6, 1> m;
		utils::ExpressionValue k;

		static IncompressibleOgdenExpr from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		IncompressibleOgden eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
