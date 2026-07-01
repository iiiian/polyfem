#pragma once

#include <polyfem/materials/Materials.hpp>
#include <polyfem/materials/MaterialExprRegistryImpl.hpp>

namespace polyfem::material
{
	/// @brief Concrete material expression registry carrying every material type
	/// defined in Materials.hpp, including all 1D/2D/3D specializations of the
	/// dim-templated materials.
	///
	/// ## Cheatsheet
	///
	/// MaterialExprRegistry r;
	///
	/// // Query material existence.
	/// int element_id = 56;
	/// bool v = r.has_material<Density<utils::ExpressionValue>>(element_id);
	/// bool v = r.has_material<Density>(element_id); // Also works.
	///
	/// // Get material. nullptr if missing.
	/// auto m = r.get<Density<utils::ExpressionValue>>(element_id);
	/// auto m = r.get<Density>(element_id);  // Also works.
	///
	/// // Set material.
	/// Density<utils::ExpressionValue> density_expr;
	/// r.set(element_id, density_expr);
	///
	/// // Remove material.
	/// r.remove<Density<utils::ExpressionValue>>(element_id);
	/// r.remove<Density>(element_id);  // Also works.
	using MaterialExprRegistry = MaterialExprRegistryImpl<
		Density,
		LinearElasticity,
		HookeLinearElasticity1D,
		HookeLinearElasticity2D,
		HookeLinearElasticity3D,
		SaintVenant1D,
		SaintVenant2D,
		SaintVenant3D,
		NeoHookean,
		IsochoricNeoHookean,
		IncompressibleLinearElasticity,
		FixedCorotational,
		MooneyRivlin,
		MooneyRivlin3Param,
		MooneyRivlin3ParamSymbolic,
		UnconstrainedOgden,
		IncompressibleOgden,
		Stokes,
		NavierStokes,
		OperatorSplitting,
		Electrostatics,
		Helmholtz,
		VolumePenalty,
		HGOFiber1D,
		HGOFiber2D,
		HGOFiber3D,
		ActiveFiber1D,
		ActiveFiber2D,
		ActiveFiber3D,
		AMIPS>;
} // namespace polyfem::material
