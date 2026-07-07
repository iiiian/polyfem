#pragma once

#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/ElementBases.hpp>

namespace polyfem::assembler
{

	/// Append cached assembly data for one element.
	///
	/// @tparam dim Element dimension.
	/// @param bases Solution basis data.
	/// @param geom_bases Geometry basis data.
	/// @param element_id Element id.
	/// @param is_mass True for mass matrix assembler.
	/// @param temp Reusable temporary storage / assembly cache output.
	template <int dim>
	void compute_assembly_cache_single(
		const ElementBasesView &bases,
		const ElementBasesView &geom_bases,
		int element_id,
		bool is_mass,
		AssemblyTempStorage &temp);

	/// Build assembly cache for all elements.
	///
	/// @param bases Solution basis data.
	/// @param geom_bases Geometry basis data.
	/// @param is_mass True for mass matrix assembler.
	AssemblyCache compute_assembly_cache_batched(
		const ElementBasesView &bases,
		const ElementBasesView &geom_bases,
		bool is_mass);

} // namespace polyfem::assembler
