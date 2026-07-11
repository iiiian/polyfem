#pragma once

#include <polyfem/materials/Dummy.hpp>
#include <polyfem/materials/LinearElasticity.hpp>
#include <polyfem/materials/Density.hpp>
#include <polyfem/materials/HookeLinearElasticity.hpp>
#include <polyfem/materials/SaintVenant.hpp>
#include <polyfem/materials/NeoHookean.hpp>
#include <polyfem/materials/IsochoricNeoHookean.hpp>
#include <polyfem/materials/IncompressibleLinearElasticity.hpp>
#include <polyfem/materials/FixedCorotational.hpp>
#include <polyfem/materials/MooneyRivlin.hpp>
#include <polyfem/materials/MooneyRivlin3Param.hpp>
#include <polyfem/materials/MooneyRivlin3ParamSymbolic.hpp>
#include <polyfem/materials/UnconstrainedOgden.hpp>
#include <polyfem/materials/IncompressibleOgden.hpp>
#include <polyfem/materials/Stokes.hpp>
#include <polyfem/materials/NavierStokes.hpp>
#include <polyfem/materials/OperatorSplitting.hpp>
#include <polyfem/materials/Electrostatics.hpp>
#include <polyfem/materials/Helmholtz.hpp>
#include <polyfem/materials/VolumePenalty.hpp>
#include <polyfem/materials/HGOFiber.hpp>
#include <polyfem/materials/ActiveFiber.hpp>
#include <polyfem/materials/AMIPS.hpp>
#include <polyfem/materials/MaterialExprRegistryImpl.hpp>

namespace polyfem::material
{
	/// @brief Concrete material expression registry carrying every material
	/// expression type, including all 1D/2D/3D material variants.
	///
	/// ## Cheatsheet
	///
	/// MaterialExprRegistry r;
	///
	/// // Query material existence.
	/// int element_id = 56;
	/// bool v = r.has_material<DensityExpr>(element_id);
	///
	/// // Get material. nullptr if missing.
	/// auto m = r.get<DensityExpr>(element_id);
	///
	/// // Set material.
	/// DensityExpr density_expr;
	/// r.set(element_id, density_expr);
	///
	/// // Remove material.
	/// r.remove<DensityExpr>(element_id);
	using MaterialExprRegistry = MaterialExprRegistryImpl<
		DensityExpr,
		LinearElasticityExpr,
		HookeLinearElasticityExpr<1>,
		HookeLinearElasticityExpr<2>,
		HookeLinearElasticityExpr<3>,
		SaintVenantExpr<1>,
		SaintVenantExpr<2>,
		SaintVenantExpr<3>,
		NeoHookeanExpr,
		IsochoricNeoHookeanExpr,
		IncompressibleLinearElasticityExpr,
		FixedCorotationalExpr,
		MooneyRivlinExpr,
		MooneyRivlin3ParamExpr,
		MooneyRivlin3ParamSymbolicExpr,
		UnconstrainedOgdenExpr,
		IncompressibleOgdenExpr,
		StokesExpr,
		NavierStokesExpr,
		OperatorSplittingExpr,
		ElectrostaticsExpr,
		HelmholtzExpr,
		VolumePenaltyExpr,
		HGOFiberExpr<1>,
		HGOFiberExpr<2>,
		HGOFiberExpr<3>,
		ActiveFiberExpr<1>,
		ActiveFiberExpr<2>,
		ActiveFiberExpr<3>,
		AMIPSExpr>;
} // namespace polyfem::material
