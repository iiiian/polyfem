#pragma once

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>
#include <functional>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#endif

namespace polyfem::basis
{

	enum class ElementKind
	{
		Simplex,
		Quad,
		Hex,
		Prism,
		Pyramid,
		Polygon,
		Polyhedron
	};

	enum class BasisFamily
	{
		Unknown,  // Placeholder for basis relying on opaque callback
		Lagrange, // This represents lagrange + berstein.
		Rational

		// QuadraticSpline,
		// PolyRbf,
		// Barycentric
	};

	/// Callback to eval basis value and gradient.
	///
	/// Let N = element basis num.
	///     D = element dimension. Can be 1,2, or 3.
	///     Q = quadrature num.
	/// quad_x/y/z -> size Q quadrature points span. y is empty for D<2. z is empty for D<2.
	/// values     -> size N*Q basis value output.
	///               Layout: [ basis0(q0) basis1(q1) ... ] [ basis1(q0) basis1(q1) ...] ...
	/// grad_x/y/z -> size N*Q basis gradient output. y is empty for D<2. z is empty for D<2.
	///               Layout: [ basis0(q0) basis1(q1) ... ] [ basis1(q0) basis1(q1) ...] ...
	// clang-format off
	using BasisEvalCallback = std::function<void(Span<const double> quad_x,
			                                         Span<const double> quad_y,
			                                         Span<const double> quad_z,
			                                         Span<double> values,
			                                         Span<double> grad_x,
			                                         Span<double> grad_y,
			                                         Span<double> grad_z)>;
	// clang-format on

	struct BasisDesc
	{
		// Common
		ElementKind element_kind;
		BasisFamily basis_family;
		int order;
		int orderq;
		int dim;
		int basis_num;
		int eval_callback_id;
		// For poly basis, both quad point and basis grad lives in physical space.
		// For others, they lives in reference space.

		// Lagrange
		bool is_bernstein;

		// Rational
		Range rational_weight_range;
	};

	struct BasisStoreView
	{
		Span<const double> rational_weights;
		Span<const BasisEvalCallback> eval_callbacks;
	};

	class BasisStore
	{
	private:
		std::vector<double> rational_weights_;
		std::vector<BasisEvalCallback> eval_callbacks_;

#ifdef POLYFEM_WITH_CUDA
		DBuf<double> d_rational_weights_;
		// BasisEvalCallback can not be used on device.
#endif

	public:
		Range append_rational_weights(Span<const double> weights);
		int append_eval_callback(BasisEvalCallback callback);
		BasisStoreView view() const;

#ifdef POLYFEM_WITH_CUDA
		bool need_host_device_sync_ = true;

		/// Return view on device memory. Lazily sync data.
		BasisStoreView device_view(CudaExecutionPolicy policy = {});

		/// Release device storage.
		void clear_device_storage();
#endif
	};

} // namespace polyfem::basis
