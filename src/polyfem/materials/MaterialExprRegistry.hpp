#pragma once

#include <polyfem/materials/Dummy.hpp>
#include <polyfem/materials/Density.hpp>
#include <polyfem/materials/NeoHookean.hpp>
#include <polyfem/materials/MaterialExprRegistryImpl.hpp>

namespace polyfem::material
{
	/// @brief Concrete material expression registry for the currently supported
	/// density and Neo-Hookean material expressions.
	///
	/// ## Cheatsheet
	///
	/// MaterialExprRegistry r;
	///
	/// // Query material existence.
	/// int element_id = 56;
	/// bool v = r.has_material<Density<double>::ExprType>(element_id);
	///
	/// // Get material. nullptr if missing.
	/// auto m = r.get<Density<double>::ExprType>(element_id);
	///
	/// // Set material.
	/// Density<double>::ExprType density_expr;
	/// r.set(element_id, density_expr);
	///
	/// // Remove material.
	/// r.remove<Density<double>::ExprType>(element_id);
	using MaterialExprRegistry = MaterialExprRegistryImpl<
		Density<double>::ExprType,
		NeoHookean<double>::ExprType>;
} // namespace polyfem::material
