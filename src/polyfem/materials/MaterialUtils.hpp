#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/materials/LameParameter.hpp>
#include <polyfem/materials/ElasticityTensor.hpp>
#include <polyfem/materials/FiberDirection.hpp>

#include <Eigen/Core>

#include <string>

#include <spdlog/fmt/fmt.h>

#include <cmath>
#include <stdexcept>

namespace polyfem::material
{
	inline utils::ExpressionValue parse_expr(const json &value, const std::string &unit_type, const std::string &root_path)
	{
		utils::ExpressionValue out;
		out.init(value, root_path);
		out.set_unit_type(unit_type);
		return out;
	}

	inline Eigen::Matrix<utils::ExpressionValue, Eigen::Dynamic, 1, 0, 6, 1> parse_ogden_vector(
		const json &value,
		const std::string &unit_type,
		const std::string &root_path)
	{
		bool is_array = value.is_array();
		int size = is_array ? value.size() : 1;
		if (size == 0)
			throw std::runtime_error("Ogden vector: expected at least one value");
		if (size > 6)
			throw std::runtime_error(fmt::format("Ogden vector: expected at most 6 values, got {}", size));
		Eigen::Matrix<utils::ExpressionValue, Eigen::Dynamic, 1, 0, 6, 1> out;
		out.resize(size);
		for (std::size_t i = 0; i < size; ++i)
			out(i) = parse_expr(is_array ? value[i] : value, unit_type, root_path);
		return out;
	}

	inline Eigen::Matrix<double, 6, 6, Eigen::RowMajor> stiffness_rotation_voigt_3d(const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> &rot)
	{
		static const double sqrt2 = std::sqrt(2.0);
		Eigen::Matrix<double, 6, 6, Eigen::RowMajor> res;
		res << rot(0, 0) * rot(0, 0), rot(0, 1) * rot(0, 1), rot(0, 2) * rot(0, 2), sqrt2 * rot(0, 1) * rot(0, 2), sqrt2 * rot(0, 0) * rot(0, 2), sqrt2 * rot(0, 0) * rot(0, 1),
			rot(1, 0) * rot(1, 0), rot(1, 1) * rot(1, 1), rot(1, 2) * rot(1, 2), sqrt2 * rot(1, 1) * rot(1, 2), sqrt2 * rot(1, 0) * rot(1, 2), sqrt2 * rot(1, 0) * rot(1, 1),
			rot(2, 0) * rot(2, 0), rot(2, 1) * rot(2, 1), rot(2, 2) * rot(2, 2), sqrt2 * rot(2, 1) * rot(2, 2), sqrt2 * rot(2, 0) * rot(2, 2), sqrt2 * rot(2, 0) * rot(2, 1),
			sqrt2 * rot(1, 0) * rot(2, 0), sqrt2 * rot(1, 1) * rot(2, 1), sqrt2 * rot(1, 2) * rot(2, 2), rot(1, 1) * rot(2, 2) + rot(1, 2) * rot(2, 1), rot(1, 0) * rot(2, 2) + rot(1, 2) * rot(2, 0), rot(1, 0) * rot(2, 1) + rot(1, 1) * rot(2, 0),
			sqrt2 * rot(0, 0) * rot(2, 0), sqrt2 * rot(0, 1) * rot(2, 1), sqrt2 * rot(0, 2) * rot(2, 2), rot(0, 1) * rot(2, 2) + rot(0, 2) * rot(2, 1), rot(0, 0) * rot(2, 2) + rot(0, 2) * rot(2, 0), rot(0, 0) * rot(2, 1) + rot(0, 1) * rot(2, 0),
			sqrt2 * rot(0, 0) * rot(1, 0), sqrt2 * rot(0, 1) * rot(1, 1), sqrt2 * rot(0, 2) * rot(1, 2), rot(0, 1) * rot(1, 2) + rot(0, 2) * rot(1, 1), rot(0, 0) * rot(1, 2) + rot(0, 2) * rot(1, 0), rot(0, 0) * rot(1, 1) + rot(0, 1) * rot(1, 0);
		return res;
	}

	template <int dim>
	inline void maybe_rotate_elasticity_tensor(const FiberDirection<double, dim> &fiber_direction, ElasticityTensor<double, dim> tensor)
	{
		if constexpr (dim == 3)
		{
			if (fiber_direction.kind == FiberDirection<double, dim>::Kind::StructureTensorOrRotation)
			{
				auto rotation = stiffness_rotation_voigt_3d(fiber_direction.direction);
				tensor.stiffness = rotation * tensor.stiffness * rotation.transpose();
			}
		}
	}

	template <typename InVector>
	inline Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 6, 1> eval_ogden_vector(const InVector &in, double x, double y, double z, double t, int element_id)
	{
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 6, 1> out;
		out.resize(in.size());
		for (int i = 0; i < in.size(); ++i)
			out(i) = in(i)(x, y, z, t, element_id);
		return out;
	}
} // namespace polyfem::material
