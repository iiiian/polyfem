#include <polyfem/materials/Materials.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/materials/LameParameter.hpp>
#include <polyfem/materials/ElasticityTensor.hpp>
#include <polyfem/materials/FiberDirection.hpp>

#include <spdlog/fmt/fmt.h>
#include <Eigen/Core>

#include <stdexcept>
#include <string>
#include <cmath>

namespace polyfem::material
{
	namespace
	{
		using Expr = utils::ExpressionValue;
		using OgdenVector = Eigen::Matrix<Expr, Eigen::Dynamic, 1, 0, 6, 1>;

		Expr parse_expr(const json &value, const std::string &unit_type, const std::string &root_path)
		{
			Expr out;
			out.init(value, root_path);
			out.set_unit_type(unit_type);
			return out;
		}

		bool parse_bool(const json &value)
		{
			return value.get<bool>();
		}

		OgdenVector parse_ogden_vector(const json &value, const std::string &unit_type, const std::string &root_path)
		{
			bool is_array = value.is_array();
			int size = is_array ? value.size() : 1;

			if (size == 0)
			{
				throw std::runtime_error("Ogden vector: expected at least one value");
			}
			if (size > 6)
			{
				throw std::runtime_error(fmt::format("Ogden vector: expected at most 6 values, got {}", size));
			}

			OgdenVector out;
			out.resize(size);
			for (std::size_t i = 0; i < size; ++i)
			{
				out(i) = parse_expr(is_array ? value[i] : value, unit_type, root_path);
			}

			return out;
		}

		Eigen::Matrix<double, 6, 6, Eigen::RowMajor> stiffness_rotation_voigt_3d(const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> &rot)
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
		void maybe_rotate_elasticity_tensor(
			const FiberDirection<double, dim> &fiber_direction,
			ElasticityTensor<double, dim> tensor)
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
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 6, 1> eval_ogden_vector(const InVector &in, double x, double y, double z, double t, int element_id)
		{
			Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 6, 1> out;
			out.resize(in.size());
			for (int i = 0; i < in.size(); ++i)
				out(i) = in(i)(x, y, z, t, element_id);
			return out;
		}
	} // namespace

	template <typename Scalar>
	auto LinearElasticity<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.lame = LameParameter<Expr>::from_json(j, units, root_path);
		return out;
	}

	// template specialization
	template LinearElasticity<Expr> LinearElasticity<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	LinearElasticity<double> LinearElasticity<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		LinearElasticity<double> out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template LinearElasticity<double> LinearElasticity<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto Density<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		if (j.contains("rho"))
		{
			out.rho = parse_expr(j.at("rho"), units.density(), root_path);
		}
		else
		{
			out.rho = parse_expr(j.at("density"), units.density(), root_path);
		}
		return out;
	}

	// template specialization
	template Density<Expr> Density<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	Density<double> Density<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Density<double> out{};
		out.rho = rho(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template Density<double> Density<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar, int dim>
	auto HookeLinearElasticityDim<Scalar, dim>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		if (j.contains("elasticity_tensor") && !j.at("elasticity_tensor").empty())
		{
			out.elasticity_tensor = ElasticityTensor<Expr, dim>::from_json(j.at("elasticity_tensor"), units, root_path);
		}
		else
		{
			out.lame = LameParameter<Expr>::from_json(j, units, root_path);
		}

		out.fiber_direction = j.contains("fiber_direction")
								  ? FiberDirection<Expr, dim>::from_json(j.at("fiber_direction"), units, root_path)
								  : FiberDirection<Expr, dim>::from_json(json::array(), units, root_path);
		return out;
	}

	// template specialization
	template HookeLinearElasticityDim<Expr, 1> HookeLinearElasticityDim<Expr, 1>::from_json(const json &, const Units &, const std::string &);
	template HookeLinearElasticityDim<Expr, 2> HookeLinearElasticityDim<Expr, 2>::from_json(const json &, const Units &, const std::string &);
	template HookeLinearElasticityDim<Expr, 3> HookeLinearElasticityDim<Expr, 3>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar, int dim>
	HookeLinearElasticityDim<double, dim> HookeLinearElasticityDim<Scalar, dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		HookeLinearElasticityDim<double, dim> out{};
		out.fiber_direction = fiber_direction.eval_expr(x, y, z, t, element_id);

		if (elasticity_tensor.depends_on_elastic_coeff)
		{
			out.lame = lame.eval_expr(x, y, z, t, element_id);
			out.elasticity_tensor = ElasticityTensor<double, dim>::from_lame(out.lame);
		}
		else
		{
			out.elasticity_tensor = elasticity_tensor.eval_expr(x, y, z, t, element_id);
		}

		maybe_rotate_elasticity_tensor(out.fiber_direction, out.elasticity_tensor);
		return out;
	}

	// template specialization
	template HookeLinearElasticityDim<double, 1> HookeLinearElasticityDim<Expr, 1>::eval_expr(double, double, double, double, int) const;
	template HookeLinearElasticityDim<double, 2> HookeLinearElasticityDim<Expr, 2>::eval_expr(double, double, double, double, int) const;
	template HookeLinearElasticityDim<double, 3> HookeLinearElasticityDim<Expr, 3>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar, int dim>
	auto SaintVenantDim<Scalar, dim>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		if (j.contains("elasticity_tensor"))
		{
			out.elasticity_tensor = ElasticityTensor<Expr, dim>::from_json(j.at("elasticity_tensor"), units, root_path);
		}
		else
		{
			out.lame = LameParameter<Expr>::from_json(j, units, root_path);
		}

		out.fiber_direction = j.contains("fiber_direction")
								  ? FiberDirection<Expr, dim>::from_json(j.at("fiber_direction"), units, root_path)
								  : FiberDirection<Expr, dim>::from_json(json::array(), units, root_path);
		return out;
	}

	// template specialization
	template SaintVenantDim<Expr, 1> SaintVenantDim<Expr, 1>::from_json(const json &, const Units &, const std::string &);
	template SaintVenantDim<Expr, 2> SaintVenantDim<Expr, 2>::from_json(const json &, const Units &, const std::string &);
	template SaintVenantDim<Expr, 3> SaintVenantDim<Expr, 3>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar, int dim>
	SaintVenantDim<double, dim> SaintVenantDim<Scalar, dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		SaintVenantDim<double, dim> out{};
		out.fiber_direction = fiber_direction.eval_expr(x, y, z, t, element_id);

		if (elasticity_tensor.depends_on_elastic_coeff)
		{
			out.lame = lame.eval_expr(x, y, z, t, element_id);
			out.elasticity_tensor = ElasticityTensor<double, dim>::from_lame(out.lame);
		}
		else
		{
			out.elasticity_tensor = elasticity_tensor.eval_expr(x, y, z, t, element_id);
		}

		maybe_rotate_elasticity_tensor(out.fiber_direction, out.elasticity_tensor);
		return out;
	}

	// template specialization
	template SaintVenantDim<double, 1> SaintVenantDim<Expr, 1>::eval_expr(double, double, double, double, int) const;
	template SaintVenantDim<double, 2> SaintVenantDim<Expr, 2>::eval_expr(double, double, double, double, int) const;
	template SaintVenantDim<double, 3> SaintVenantDim<Expr, 3>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto NeoHookean<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.lame = LameParameter<Expr>::from_json(j, units, root_path);
		return out;
	}

	// template specialization
	template NeoHookean<Expr> NeoHookean<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	NeoHookean<double> NeoHookean<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		NeoHookean<double> out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template NeoHookean<double> NeoHookean<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto IsochoricNeoHookean<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.lame = LameParameter<Expr>::from_json(j, units, root_path);
		return out;
	}

	// template specialization
	template IsochoricNeoHookean<Expr> IsochoricNeoHookean<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	IsochoricNeoHookean<double> IsochoricNeoHookean<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		IsochoricNeoHookean<double> out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template IsochoricNeoHookean<double> IsochoricNeoHookean<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto IncompressibleLinearElasticity<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.lame = LameParameter<Expr>::from_json(j, units, root_path);
		return out;
	}

	// template specialization
	template IncompressibleLinearElasticity<Expr> IncompressibleLinearElasticity<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	IncompressibleLinearElasticity<double> IncompressibleLinearElasticity<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		IncompressibleLinearElasticity<double> out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template IncompressibleLinearElasticity<double> IncompressibleLinearElasticity<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto FixedCorotational<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.lame = LameParameter<Expr>::from_json(j, units, root_path);
		return out;
	}

	// template specialization
	template FixedCorotational<Expr> FixedCorotational<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	FixedCorotational<double> FixedCorotational<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		FixedCorotational<double> out{};
		out.lame = lame.eval_expr(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template FixedCorotational<double> FixedCorotational<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto MooneyRivlin<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		const std::string stress = units.stress();
		ExprType out;
		out.c1 = parse_expr(j.at("c1"), stress, root_path);
		out.c2 = parse_expr(j.at("c2"), stress, root_path);
		out.k = parse_expr(j.at("k"), stress, root_path);
		return out;
	}

	// template specialization
	template MooneyRivlin<Expr> MooneyRivlin<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	MooneyRivlin<double> MooneyRivlin<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		MooneyRivlin<double> out{};
		out.c1 = c1(x, y, z, t, element_id);
		out.c2 = c2(x, y, z, t, element_id);
		out.k = k(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template MooneyRivlin<double> MooneyRivlin<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto MooneyRivlin3Param<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		const std::string stress = units.stress();
		ExprType out;
		out.c1 = parse_expr(j.at("c1"), stress, root_path);
		out.c2 = parse_expr(j.at("c2"), stress, root_path);
		out.c3 = parse_expr(j.at("c3"), stress, root_path);
		out.d1 = parse_expr(j.at("d1"), stress, root_path);
		return out;
	}

	// template specialization
	template MooneyRivlin3Param<Expr> MooneyRivlin3Param<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	MooneyRivlin3Param<double> MooneyRivlin3Param<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		MooneyRivlin3Param<double> out{};
		out.c1 = c1(x, y, z, t, element_id);
		out.c2 = c2(x, y, z, t, element_id);
		out.c3 = c3(x, y, z, t, element_id);
		out.d1 = d1(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template MooneyRivlin3Param<double> MooneyRivlin3Param<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto MooneyRivlin3ParamSymbolic<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		const std::string stress = units.stress();
		ExprType out;
		out.c1 = parse_expr(j.at("c1"), stress, root_path);
		out.c2 = parse_expr(j.at("c2"), stress, root_path);
		out.c3 = parse_expr(j.at("c3"), stress, root_path);
		out.d1 = parse_expr(j.at("d1"), stress, root_path);
		return out;
	}

	// template specialization
	template MooneyRivlin3ParamSymbolic<Expr> MooneyRivlin3ParamSymbolic<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	MooneyRivlin3ParamSymbolic<double> MooneyRivlin3ParamSymbolic<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		MooneyRivlin3ParamSymbolic<double> out{};
		out.c1 = c1(x, y, z, t, element_id);
		out.c2 = c2(x, y, z, t, element_id);
		out.c3 = c3(x, y, z, t, element_id);
		out.d1 = d1(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template MooneyRivlin3ParamSymbolic<double> MooneyRivlin3ParamSymbolic<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto UnconstrainedOgden<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		const std::string stress = units.stress();
		ExprType out;
		out.alphas = parse_ogden_vector(j.at("alphas"), "", root_path);
		out.mus = parse_ogden_vector(j.at("mus"), stress, root_path);
		out.Ds = parse_ogden_vector(j.at("Ds"), stress, root_path);
		return out;
	}

	// template specialization
	template UnconstrainedOgden<Expr> UnconstrainedOgden<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	UnconstrainedOgden<double> UnconstrainedOgden<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		UnconstrainedOgden<double> out{};
		out.alphas = eval_ogden_vector(alphas, x, y, z, t, element_id);
		out.mus = eval_ogden_vector(mus, x, y, z, t, element_id);
		out.Ds = eval_ogden_vector(Ds, x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template UnconstrainedOgden<double> UnconstrainedOgden<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto IncompressibleOgden<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		const std::string stress = units.stress();
		ExprType out;
		out.c = parse_ogden_vector(j.at("c"), stress, root_path);
		out.m = parse_ogden_vector(j.at("m"), "", root_path);
		out.k = parse_expr(j.at("k"), stress, root_path);
		return out;
	}

	// template specialization
	template IncompressibleOgden<Expr> IncompressibleOgden<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	IncompressibleOgden<double> IncompressibleOgden<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		IncompressibleOgden<double> out{};
		out.c = eval_ogden_vector(c, x, y, z, t, element_id);
		out.m = eval_ogden_vector(m, x, y, z, t, element_id);
		out.k = k(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template IncompressibleOgden<double> IncompressibleOgden<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto Stokes<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.viscosity = parse_expr(j.at("viscosity"), units.viscosity(), root_path);
		return out;
	}

	// template specialization
	template Stokes<Expr> Stokes<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	Stokes<double> Stokes<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Stokes<double> out{};
		out.viscosity = viscosity(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template Stokes<double> Stokes<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto NavierStokes<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.viscosity = parse_expr(j.at("viscosity"), units.viscosity(), root_path);
		return out;
	}

	// template specialization
	template NavierStokes<Expr> NavierStokes<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	NavierStokes<double> NavierStokes<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		NavierStokes<double> out{};
		out.viscosity = viscosity(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template NavierStokes<double> NavierStokes<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto OperatorSplitting<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.viscosity = parse_expr(j.at("viscosity"), units.viscosity(), root_path);
		return out;
	}

	// template specialization
	template OperatorSplitting<Expr> OperatorSplitting<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	OperatorSplitting<double> OperatorSplitting<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		OperatorSplitting<double> out{};
		out.viscosity = viscosity(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template OperatorSplitting<double> OperatorSplitting<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto Electrostatics<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.epsilon = parse_expr(j.at("epsilon"), units.permittivity(), root_path);
		return out;
	}

	// template specialization
	template Electrostatics<Expr> Electrostatics<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	Electrostatics<double> Electrostatics<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Electrostatics<double> out{};
		out.epsilon = epsilon(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template Electrostatics<double> Electrostatics<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto Helmholtz<Scalar>::from_json(const json &j, const Units &, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.k = parse_expr(j.at("k"), "", root_path);
		return out;
	}

	// template specialization
	template Helmholtz<Expr> Helmholtz<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	Helmholtz<double> Helmholtz<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		Helmholtz<double> out{};
		out.k = k(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template Helmholtz<double> Helmholtz<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto VolumePenalty<Scalar>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.k = parse_expr(j.at("k"), units.stress(), root_path);
		return out;
	}

	// template specialization
	template VolumePenalty<Expr> VolumePenalty<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	VolumePenalty<double> VolumePenalty<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		VolumePenalty<double> out{};
		out.k = k(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template VolumePenalty<double> VolumePenalty<Expr>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar, int dim>
	auto HGOFiberDim<Scalar, dim>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		const std::string stress = units.stress();
		ExprType out;
		out.k1 = parse_expr(j.at("k1"), stress, root_path);
		out.k2 = parse_expr(j.at("k2"), "", root_path);
		out.fiber_direction = j.contains("fiber_direction")
								  ? FiberDirection<Expr, dim>::from_json(j.at("fiber_direction"), units, root_path)
								  : FiberDirection<Expr, dim>::from_json(json::array(), units, root_path);
		return out;
	}

	// template specialization
	template HGOFiberDim<Expr, 1> HGOFiberDim<Expr, 1>::from_json(const json &, const Units &, const std::string &);
	template HGOFiberDim<Expr, 2> HGOFiberDim<Expr, 2>::from_json(const json &, const Units &, const std::string &);
	template HGOFiberDim<Expr, 3> HGOFiberDim<Expr, 3>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar, int dim>
	HGOFiberDim<double, dim> HGOFiberDim<Scalar, dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		HGOFiberDim<double, dim> out{};
		out.k1 = k1(x, y, z, t, element_id);
		out.k2 = k2(x, y, z, t, element_id);
		out.fiber_direction = fiber_direction.eval_expr(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template HGOFiberDim<double, 1> HGOFiberDim<Expr, 1>::eval_expr(double, double, double, double, int) const;
	template HGOFiberDim<double, 2> HGOFiberDim<Expr, 2>::eval_expr(double, double, double, double, int) const;
	template HGOFiberDim<double, 3> HGOFiberDim<Expr, 3>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar, int dim>
	auto ActiveFiberDim<Scalar, dim>::from_json(const json &j, const Units &units, const std::string &root_path) -> ExprType
	{
		const std::string stress = units.stress();
		ExprType out;
		out.activation = parse_expr(j.at("activation"), "", root_path);
		out.Tmax = parse_expr(j.at("Tmax"), stress, root_path);
		out.fiber_direction = j.contains("fiber_direction")
								  ? FiberDirection<Expr, dim>::from_json(j.at("fiber_direction"), units, root_path)
								  : FiberDirection<Expr, dim>::from_json(json::array(), units, root_path);
		return out;
	}

	// template specialization
	template ActiveFiberDim<Expr, 1> ActiveFiberDim<Expr, 1>::from_json(const json &, const Units &, const std::string &);
	template ActiveFiberDim<Expr, 2> ActiveFiberDim<Expr, 2>::from_json(const json &, const Units &, const std::string &);
	template ActiveFiberDim<Expr, 3> ActiveFiberDim<Expr, 3>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar, int dim>
	ActiveFiberDim<double, dim> ActiveFiberDim<Scalar, dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		ActiveFiberDim<double, dim> out{};
		out.activation = activation(x, y, z, t, element_id);
		out.Tmax = Tmax(x, y, z, t, element_id);
		out.fiber_direction = fiber_direction.eval_expr(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template ActiveFiberDim<double, 1> ActiveFiberDim<Expr, 1>::eval_expr(double, double, double, double, int) const;
	template ActiveFiberDim<double, 2> ActiveFiberDim<Expr, 2>::eval_expr(double, double, double, double, int) const;
	template ActiveFiberDim<double, 3> ActiveFiberDim<Expr, 3>::eval_expr(double, double, double, double, int) const;

	template <typename Scalar>
	auto AMIPS<Scalar>::from_json(const json &j, const Units &, const std::string &root_path) -> ExprType
	{
		ExprType out;
		out.use_rest_pose = parse_bool(j.at("use_rest_pose"));
		out.weight = parse_expr(j.at("weight"), "", root_path);
		return out;
	}

	// template specialization
	template AMIPS<Expr> AMIPS<Expr>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar>
	AMIPS<double> AMIPS<Scalar>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		AMIPS<double> out{};
		out.use_rest_pose = use_rest_pose;
		out.weight = weight(x, y, z, t, element_id);
		return out;
	}

	// template specialization
	template AMIPS<double> AMIPS<Expr>::eval_expr(double, double, double, double, int) const;
} // namespace polyfem::material
