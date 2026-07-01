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
	using Expr = utils::ExpressionValue;

	// For weak form that does not require any material.
	template <typename Scalar>
	struct Dummy
	{
	};

	template <typename Scalar>
	struct LinearElasticity
	{
		using ExprType = LinearElasticity<Expr>;

		LameParameter<Scalar> lame;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		LinearElasticity<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct Density
	{
		using ExprType = Density<Expr>;

		Scalar rho;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		Density<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar, int dim>
	struct HookeLinearElasticityDim
	{
		using ExprType = HookeLinearElasticityDim<Expr, dim>;

		LameParameter<Scalar> lame;
		ElasticityTensor<Scalar, dim> elasticity_tensor;
		FiberDirection<Scalar, dim> fiber_direction;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		HookeLinearElasticityDim<double, dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	using HookeLinearElasticity1D = HookeLinearElasticityDim<Scalar, 1>;

	template <typename Scalar>
	using HookeLinearElasticity2D = HookeLinearElasticityDim<Scalar, 2>;

	template <typename Scalar>
	using HookeLinearElasticity3D = HookeLinearElasticityDim<Scalar, 3>;

	template <typename Scalar, int dim>
	struct SaintVenantDim
	{
		using ExprType = SaintVenantDim<Expr, dim>;

		LameParameter<Scalar> lame;
		ElasticityTensor<Scalar, dim> elasticity_tensor;
		FiberDirection<Scalar, dim> fiber_direction;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		SaintVenantDim<double, dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	using SaintVenant1D = SaintVenantDim<Scalar, 1>;

	template <typename Scalar>
	using SaintVenant2D = SaintVenantDim<Scalar, 2>;

	template <typename Scalar>
	using SaintVenant3D = SaintVenantDim<Scalar, 3>;

	template <typename Scalar>
	struct NeoHookean
	{
		using ExprType = NeoHookean<Expr>;

		LameParameter<Scalar> lame;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		NeoHookean<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct IsochoricNeoHookean
	{
		using ExprType = IsochoricNeoHookean<Expr>;

		LameParameter<Scalar> lame;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		IsochoricNeoHookean<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct IncompressibleLinearElasticity
	{
		using ExprType = IncompressibleLinearElasticity<Expr>;

		LameParameter<Scalar> lame;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		IncompressibleLinearElasticity<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct FixedCorotational
	{
		using ExprType = FixedCorotational<Expr>;

		LameParameter<Scalar> lame;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		FixedCorotational<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct MooneyRivlin
	{
		using ExprType = MooneyRivlin<Expr>;

		Scalar c1;
		Scalar c2;
		Scalar k;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		MooneyRivlin<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct MooneyRivlin3Param
	{
		using ExprType = MooneyRivlin3Param<Expr>;

		Scalar c1;
		Scalar c2;
		Scalar c3;
		Scalar d1;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		MooneyRivlin3Param<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct MooneyRivlin3ParamSymbolic
	{
		using ExprType = MooneyRivlin3ParamSymbolic<Expr>;

		Scalar c1;
		Scalar c2;
		Scalar c3;
		Scalar d1;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		MooneyRivlin3ParamSymbolic<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct UnconstrainedOgden
	{
		using ExprType = UnconstrainedOgden<Expr>;

		// Max order 6.
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, 6, 1> alphas;
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, 6, 1> mus;
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, 6, 1> Ds;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		UnconstrainedOgden<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct IncompressibleOgden
	{
		using ExprType = IncompressibleOgden<Expr>;

		// Max order 6.
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, 6, 1> c;
		Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, 6, 1> m;
		Scalar k;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		IncompressibleOgden<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct Stokes
	{
		using ExprType = Stokes<Expr>;

		Scalar viscosity;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		Stokes<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct NavierStokes
	{
		using ExprType = NavierStokes<Expr>;

		Scalar viscosity;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		NavierStokes<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct OperatorSplitting
	{
		using ExprType = OperatorSplitting<Expr>;

		Scalar viscosity;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		OperatorSplitting<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct Electrostatics
	{
		using ExprType = Electrostatics<Expr>;

		Scalar epsilon;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		Electrostatics<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct Helmholtz
	{
		using ExprType = Helmholtz<Expr>;

		Scalar k;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		Helmholtz<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	struct VolumePenalty
	{
		using ExprType = VolumePenalty<Expr>;

		Scalar k;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		VolumePenalty<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar, int dim>
	struct HGOFiberDim
	{
		using ExprType = HGOFiberDim<Expr, dim>;

		Scalar k1;
		Scalar k2;
		FiberDirection<Scalar, dim> fiber_direction;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		HGOFiberDim<double, dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	using HGOFiber1D = HGOFiberDim<Scalar, 1>;

	template <typename Scalar>
	using HGOFiber2D = HGOFiberDim<Scalar, 2>;

	template <typename Scalar>
	using HGOFiber3D = HGOFiberDim<Scalar, 3>;

	template <typename Scalar, int dim>
	struct ActiveFiberDim
	{
		using ExprType = ActiveFiberDim<Expr, dim>;

		Scalar activation;
		Scalar Tmax;
		FiberDirection<Scalar, dim> fiber_direction;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		ActiveFiberDim<double, dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

	template <typename Scalar>
	using ActiveFiber1D = ActiveFiberDim<Scalar, 1>;

	template <typename Scalar>
	using ActiveFiber2D = ActiveFiberDim<Scalar, 2>;

	template <typename Scalar>
	using ActiveFiber3D = ActiveFiberDim<Scalar, 3>;

	template <typename Scalar>
	struct AMIPS
	{
		using ExprType = AMIPS<Expr>;

		bool use_rest_pose;
		Scalar weight;

		static ExprType from_json(const json &j, const Units &units, const std::string &root_path);
		/// Evaluate material expression at position (x,y,z), time t, and element id.
		AMIPS<double> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};

} // namespace polyfem::material
