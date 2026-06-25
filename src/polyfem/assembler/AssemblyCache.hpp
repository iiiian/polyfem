#pragma once

#include <polyfem/basis/ElementBases.hpp>

#include <vector>

namespace polyfem::assembler
{
	/// Assembly cache descriptor for each element.
	struct AssemblyCacheDesc
	{
		bool is_empty = true;
		bool is_mass = false;

		Range basis_val_range;
		Range basis_grad_x_range;
		Range basis_grad_y_range;
		Range basis_grad_z_range;
		Range basis_grad_phy_x_range;
		Range basis_grad_phy_y_range;
		Range basis_grad_phy_z_range;

		Range physical_x_range;
		Range physical_y_range;
		Range physical_z_range;

		Range det_J_range;
		Range J_inverse_transpose_range;
		Range weighted_measure_range;
	};

	/// Non-owning view of assembly cache.
	struct AssemblyCacheView
	{
		span<const AssemblyCacheDesc> desc;

		/// Stacked basis values ϕ(q) for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ϕ0(q0) ϕ0(q1) ... ] [ ϕ1(q0) ϕ1(q1) ... ] ...
		span<const double> basis_values;
		/// Stacked raw x-gradients for all cached elements.
		/// These are in the basis evaluation coordinates: parametric/reference for
		/// parameterized bases, physical for non-parametric bases.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		span<const double> basis_grad_x;
		/// Stacked raw y-gradients for all cached elements.
		/// These are in the basis evaluation coordinates: parametric/reference for
		/// parameterized bases, physical for non-parametric bases.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		span<const double> basis_grad_y;
		/// Stacked raw z-gradients for all cached elements.
		/// These are in the basis evaluation coordinates: parametric/reference for
		/// parameterized bases, physical for non-parametric bases.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		span<const double> basis_grad_z;
		/// Stacked physical x-gradients ∂ϕ/∂x for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		span<const double> basis_grad_phy_x;
		/// Stacked physical y-gradients ∂ϕ/∂y for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		span<const double> basis_grad_phy_y;
		/// Stacked physical z-gradients ∂ϕ/∂z for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		span<const double> basis_grad_phy_z;

		/// Physical x-coordinate x(q) for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ x(q0) x(q1) ... ].
		span<const double> physical_x;
		/// Physical y-coordinate y(q) for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ y(q0) y(q1) ... ].
		span<const double> physical_y;
		/// Physical z-coordinate z(q) for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ z(q0) z(q1) ... ].
		span<const double> physical_z;

		/// Determinant det(J(q)) of the reference-to-physical jacobian for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ det(J(q0)) det(J(q1)) ... ].
		span<const double> det_J;
		/// Inverse-transpose J(q)^{-T} of the reference-to-physical jacobian for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ row-major J(q0)^{-T} ] [ row-major J(q1)^{-T} ] ...
		span<const double> J_inverse_transpose;
		/// quad_weight(q)*det(J(q)) using the reference-to-physical jacobian for all cached elements.
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ w(q0)det(J(q0)) w(q1)det(J(q1)) ... ].
		span<const double> weighted_measure;
	};

	/// Storage for basis evaluation, physical position, and geometry mapping.
	/// See AssemblyCacheView for data layout;
	struct AssemblyCache
	{
		std::vector<AssemblyCacheDesc> desc;

		std::vector<double> basis_values;
		std::vector<double> basis_grad_x;
		std::vector<double> basis_grad_y;
		std::vector<double> basis_grad_z;
		std::vector<double> basis_grad_phy_x;
		std::vector<double> basis_grad_phy_y;
		std::vector<double> basis_grad_phy_z;

		std::vector<double> physical_x;
		std::vector<double> physical_y;
		std::vector<double> physical_z;

		std::vector<double> det_J;
		std::vector<double> J_inverse_transpose;
		std::vector<double> weighted_measure;

		AssemblyCacheView view() const;

		// Resize all storage to zero without actually freeing the memory.
		void clear();
	};

	/// Temporary storage for intermediate value during geometry mapping evaluation.
	struct GeomMappingScratch
	{
		std::vector<double> basis_values;
		std::vector<double> basis_grad_x;
		std::vector<double> basis_grad_y;
		std::vector<double> basis_grad_z;
	};

	/// Append cached assembly data for one element.
	///
	/// @param bases Solution basis data.
	/// @param geom_bases Geometry basis data.
	/// @param element_id Element id.
	/// @param is_mass True for mass matrix assembler.
	/// @param is_isoparametric True if geometry and solution bases are the same.
	/// @param need_basis_values If true, eval solution basis values.
	/// @param need_basis_gradients If ture, eval solution basis gradients.
	/// @param geom_mapping_scratch Reusable temporary storage.
	/// @param cache Assembly cache to append into.
	void compute_assembly_cache_single(
		const basis::ng::ElementBasesView &bases,
		const basis::ng::ElementBasesView &geom_bases,
		int element_id,
		bool is_mass,
		bool is_isoparametric,
		bool need_basis_values,
		bool need_basis_gradients,
		GeomMappingScratch &geom_mapping_scratch,
		AssemblyCache &cache);

	/// Build assembly cache for all elements.
	///
	/// @param bases Solution basis data.
	/// @param geom_bases Geometry basis data.
	/// @param is_mass True for mass matrix assembler.
	/// @param is_isoparametric True if geometry and solution bases are the same.
	/// @param need_basis_values If true, eval solution basis values.
	/// @param need_basis_gradients If ture, eval solution basis gradients.
	AssemblyCache compute_assembly_cache_batched(
		const basis::ng::ElementBasesView &bases,
		const basis::ng::ElementBasesView &geom_bases,
		bool is_mass,
		bool is_isoparametric,
		bool need_basis_values,
		bool need_basis_gradients);

} // namespace polyfem::assembler
