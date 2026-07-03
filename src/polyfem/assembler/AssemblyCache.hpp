#pragma once

#include <polyfem/utils/Range.hpp>
#include <polyfem/utils/Span.hpp>
#include <polyfem/assembler/ElementBases.hpp>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#endif

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

	struct AssemblyCacheView
	{
		Span<const AssemblyCacheDesc> desc;

		/// Stacked basis values ϕ(q) for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ϕ0(q0) ϕ0(q1) ... ] [ ϕ1(q0) ϕ1(q1) ... ] ...
		Span<const double> basis_values;

		/// Stacked raw x-gradients for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		Span<const double> basis_grad_x;

		/// Stacked raw y-gradients for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		Span<const double> basis_grad_y;

		/// Stacked raw z-gradients for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		Span<const double> basis_grad_z;

		/// Stacked physical x-gradients ∂ϕ/∂x for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		Span<const double> basis_grad_phy_x;

		/// Stacked physical y-gradients ∂ϕ/∂y for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		Span<const double> basis_grad_phy_y;

		/// Stacked physical z-gradients ∂ϕ/∂z for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ ∂ϕ0(q0) ∂ϕ0(q1) ... ] [ ∂ϕ1(q0) ∂ϕ1(q1) ... ] ...
		Span<const double> basis_grad_phy_z;

		/// Physical x-coordinate x(q) for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ x(q0) x(q1) ... ].
		Span<const double> physical_x;

		/// Physical y-coordinate y(q) for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ y(q0) y(q1) ... ].
		Span<const double> physical_y;

		/// Physical z-coordinate z(q) for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ z(q0) z(q1) ... ].
		Span<const double> physical_z;

		/// Determinant det(J(q)) of the reference-to-physical jacobian for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ det(J(q0)) det(J(q1)) ... ].
		Span<const double> det_J;

		/// Inverse-transpose J(q)^{-T} of the reference-to-physical jacobian for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ row-major J(q0)^{-T} ] [ row-major J(q1)^{-T} ] ...
		Span<const double> J_inverse_transpose;

		/// quad_weight(q)*det(J(q)) using the reference-to-physical jacobian for all cached elements.
		///
		/// Layout: [ element 0 cache ] [ element 1 cache ] ...
		/// Layout inside each element: [ w(q0)det(J(q0)) w(q1)det(J(q1)) ... ].
		Span<const double> weighted_measure;
	};

	// Temp storage required to compute assembly cache.
	struct AssemblyTempStorage
	{
		std::vector<double> basis_values;
		std::vector<double> basis_grad_x;
		std::vector<double> basis_grad_y;
		std::vector<double> basis_grad_z;
		std::vector<double> basis_grad_phy_x;
		std::vector<double> basis_grad_phy_y;
		std::vector<double> basis_grad_phy_z;
		std::vector<double> gbasis_values;
		std::vector<double> gbasis_grad_x;
		std::vector<double> gbasis_grad_y;
		std::vector<double> gbasis_grad_z;
		std::vector<double> physical_x;
		std::vector<double> physical_y;
		std::vector<double> physical_z;
		std::vector<double> det_J;
		std::vector<double> J_inverse_transpose;
		std::vector<double> weighted_measure;

		void resize(int dim, int basis_num, int geom_basis_num, int quad_num);
	};

	/// Storage for basis evaluation, physical position, and geometry mapping.
	/// See AssemblyCacheView for data layout.
	class AssemblyCache
	{
	private:
		std::vector<AssemblyCacheDesc> desc_;

		std::vector<double> basis_values_;
		std::vector<double> basis_grad_x_;
		std::vector<double> basis_grad_y_;
		std::vector<double> basis_grad_z_;
		std::vector<double> basis_grad_phy_x_;
		std::vector<double> basis_grad_phy_y_;
		std::vector<double> basis_grad_phy_z_;

		std::vector<double> physical_x_;
		std::vector<double> physical_y_;
		std::vector<double> physical_z_;

		std::vector<double> det_J_;
		std::vector<double> J_inverse_transpose_;
		std::vector<double> weighted_measure_;

#ifdef POLYFEM_WITH_CUDA
		bool need_host_device_sync_ = true;

		DeviceBuf<AssemblyCacheDesc> d_desc_;
		DeviceBuf<double> d_basis_values_;
		DeviceBuf<double> d_basis_grad_x_;
		DeviceBuf<double> d_basis_grad_y_;
		DeviceBuf<double> d_basis_grad_z_;
		DeviceBuf<double> d_basis_grad_phy_x_;
		DeviceBuf<double> d_basis_grad_phy_y_;
		DeviceBuf<double> d_basis_grad_phy_z_;
		DeviceBuf<double> d_physical_x_;
		DeviceBuf<double> d_physical_y_;
		DeviceBuf<double> d_physical_z_;
		DeviceBuf<double> d_det_J_;
		DeviceBuf<double> d_J_inverse_transpose_;
		DeviceBuf<double> d_weighted_measure_;
#endif

	public:
		int append(bool is_mass, const AssemblyTempStorage &temp);

		AssemblyCacheView view() const;

#ifdef POLYFEM_WITH_CUDA
		AssemblyCacheView device_view(CudaExecutionPolicy policy = {});
		void clear_device_storage();
#endif
	};

} // namespace polyfem::assembler
