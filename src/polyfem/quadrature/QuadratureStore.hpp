#pragma once

#include <polyfem/quadrature/Quadrature.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#endif

namespace polyfem::quadrature
{

	struct QuadratureDesc
	{
		int dim = -1;
		Range x_range;
		Range y_range;
		Range z_range;
		Range w_range;
	};

	struct QuadratureStoreView
	{
		Span<const double> x;
		Span<const double> y;
		Span<const double> z;
		Span<const double> w;
	};

	class QuadratureStore
	{
	private:
		std::vector<double> x_; //< Quadrature point coordinate 0.
		std::vector<double> y_; //< Quadrature point coordinate 1.
		std::vector<double> z_; //< Quadrature point coordinate 2.
		std::vector<double> w_; //< Quadrature weight.

#ifdef POLYFEM_WITH_CUDA
		bool need_host_device_sync_ = true;
		DBuf<double> d_x_;
		DBuf<double> d_y_;
		DBuf<double> d_z_;
		DBuf<double> d_w_;
#endif

	public:
		QuadratureStoreView view() const;

		/// Append quadrature to the store. Require non-empty quadrature.
		QuadratureDesc append(const quadrature::Quadrature &quad);

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device memory. Lazily sync data.
		QuadratureStoreView device_view(CudaExecutionPolicy policy = {});

		/// Release device storage.
		void clear_device_storage();
#endif
	};
} // namespace polyfem::quadrature
