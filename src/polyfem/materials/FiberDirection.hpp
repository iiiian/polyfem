#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <Eigen/Core>

#include <string>

namespace polyfem::material
{
	template <typename Scalar, int dim>
	struct FiberDirection
	{
		static_assert(dim >= 1 && dim <= 3, "FiberDirection dimension must be 1, 2, or 3");

		/// Physical meaning of the fiber data:
		/// Identity -> Isometric.
		/// Direction -> A single direction a.
		/// StructureTensorOrRotation -> For fiber, this is the general structure tensor encoding
		/// continuous weighted fiber orientation. For linear elasticity this encodes a rotation
		/// matrix for stiffness tensor. So the meaning of the data DEPENDS ON THE PHYSICS TYPE!!
		/// TODO: clean up. split into different class.
		enum class Kind
		{
			Identity,
			Direction,
			StructureTensorOrRotation
		};

		/// Storage convention:
		/// - Direction: first column stores the vector, remaining entries are zero.
		/// - StructureTensorOrRotation: the full dim x dim matrix is meaningful.
		/// - Identity: not used.
		Eigen::Matrix<Scalar, dim, dim, Eigen::RowMajor> direction;
		Kind kind = Kind::Identity;

		/// Parse json and build fiber direction expression.
		static FiberDirection<utils::ExpressionValue, dim> from_json(const json &value, const Units &units, const std::string &root_path);

		/// Evaluate fiber direction expression at position (x,y,z), time t, and element id.
		FiberDirection<double, dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
