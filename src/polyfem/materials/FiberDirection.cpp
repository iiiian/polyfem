#include <polyfem/materials/FiberDirection.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <spdlog/fmt/fmt.h>
#include <Eigen/Core>

#include <stdexcept>
#include <string>

namespace polyfem::material
{
	namespace
	{
		template <int dim>
		void zero_fiber_direction(FiberDirection<utils::ExpressionValue, dim> &out)
		{
			for (int i = 0; i < dim; ++i)
			{
				for (int j = 0; j < dim; ++j)
				{
					out.direction(i, j).init(0.0);
				}
			}
		}
	} // namespace

	template <typename Scalar, int dim>
	FiberDirection<utils::ExpressionValue, dim> FiberDirection<Scalar, dim>::from_json(
		const json &value,
		const Units &units,
		const std::string &root_path)
	{
		using Kind = typename FiberDirection<utils::ExpressionValue, dim>::Kind;
		const std::string length_unit = units.length();

		if (!value.is_array())
		{
			throw std::runtime_error(fmt::format("fiber_direction: expected a JSON array"));
		}

		FiberDirection<utils::ExpressionValue, dim> out;

		// Fiber direction json encodes 3 differnt layouts:
		// Type 1. Empty array [] -> Identity. Isotropic.
		// Type 2. A dimxdim matrix. Represented as flattened json array or nested json array.
		// Type 3. A dim vector. Represented as a flat json array.

		// Type 1.
		if (value.empty())
		{
			out.kind = Kind::Identity;
			return out;
		}

		// Type 2. Nested json array.
		if (value.size() == dim && value[0].is_array())
		{
			zero_fiber_direction(out);
			for (int i = 0; i < dim; ++i)
			{
				if (!value[i].is_array() || value[i].size() != dim)
				{
					throw std::runtime_error(fmt::format("fiber_direction: expected a nested {}x{} matrix", dim, dim));
				}

				for (int j = 0; j < dim; ++j)
				{
					out.direction(i, j).init(value[i][j], root_path);
					out.direction(i, j).set_unit_type(length_unit);
				}
			}
			out.kind = Kind::StructureTensorOrRotation;
			return out;
		}

		for (int i = 0; i < value.size(); ++i)
		{
			if (value[i].is_array())
			{
				throw std::runtime_error(fmt::format("fiber_direction: must not mix scalar entries and matrix rows"));
			}
		}

		// Type 2. Flattened json array.
		if (value.size() == dim * dim && dim != 1)
		{
			zero_fiber_direction(out);
			for (int i = 0; i < dim; ++i)
			{
				for (int j = 0; j < dim; ++j)
				{
					out.direction(i, j).init(value[i * dim + j], root_path);
					out.direction(i, j).set_unit_type(length_unit);
				}
			}
			out.kind = Kind::StructureTensorOrRotation;
			return out;
		}

		// Type 3.
		if (value.size() == dim)
		{
			zero_fiber_direction(out);
			for (int i = 0; i < dim; ++i)
			{
				out.direction(i, 0).init(value[i], root_path);
				out.direction(i, 0).set_unit_type(length_unit);
			}
			out.kind = Kind::Direction;
			return out;
		}

		throw std::runtime_error(fmt::format(
			"fiber_direction: expected [], a {}-vector, a flat {}x{} matrix, or a nested {}x{} matrix; got {} entries",
			dim,
			dim,
			dim,
			dim,
			dim,
			value.size()));
	}

	// template specialization.
	template FiberDirection<utils::ExpressionValue, 1> FiberDirection<utils::ExpressionValue, 1>::from_json(const json &, const Units &, const std::string &);
	template FiberDirection<utils::ExpressionValue, 2> FiberDirection<utils::ExpressionValue, 2>::from_json(const json &, const Units &, const std::string &);
	template FiberDirection<utils::ExpressionValue, 3> FiberDirection<utils::ExpressionValue, 3>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar, int dim>
	FiberDirection<double, dim> FiberDirection<Scalar, dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		using OutType = FiberDirection<double, dim>;
		OutType out{};

		if (kind == FiberDirection<Scalar, dim>::Kind::Identity)
		{
			out.kind = OutType::Kind::Identity;
			out.direction.setIdentity();
			return out;
		}

		if (kind == FiberDirection<Scalar, dim>::Kind::Direction)
		{
			out.kind = OutType::Kind::Direction;
			for (int i = 0; i < dim; ++i)
			{
				out.direction(i, 0) = direction(i, 0)(x, y, z, t, element_id);
			}
			return out;
		}

		out.kind = OutType::Kind::StructureTensorOrRotation;
		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < dim; ++j)
			{
				out.direction(i, j) = direction(i, j)(x, y, z, t, element_id);
			}
		}
		return out;
	}

	template FiberDirection<double, 1> FiberDirection<utils::ExpressionValue, 1>::eval_expr(double, double, double, double, int) const;
	template FiberDirection<double, 2> FiberDirection<utils::ExpressionValue, 2>::eval_expr(double, double, double, double, int) const;
	template FiberDirection<double, 3> FiberDirection<utils::ExpressionValue, 3>::eval_expr(double, double, double, double, int) const;
} // namespace polyfem::material
