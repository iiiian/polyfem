#include "polyfem/utils/CUDAExecutionPolicy.hpp"
#include <polyfem/quadrature/QuadratureStore.hpp>

#include <polyfem/quadrature/Quadrature.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::quadrature
{

	QuadratureStoreView QuadratureStore::view() const
	{
		return QuadratureStoreView{x_, y_, z_, w_};
	}

	/// @brief Append quadrature to the store. Require non-empty quadrature.
	QuadratureDesc QuadratureStore::append(const quadrature::Quadrature &quad)
	{
		assert(quad.size() != 0);

#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		// Quadrature class stores:
		// - points: A matrix of size (quad_num x dim). Each row is a quadrature point.
		// - weights: A vector of size quad_num.

		QuadratureDesc desc;
		desc.dim = quad.points.cols();
		// dim == 1, y and z are empty.
		if (desc.dim == 1)
		{
			auto col_x = quad.points.col(0);
			desc.x_range = Range{static_cast<int>(x_.size()), static_cast<int>(col_x.size())};
			desc.y_range = {};
			desc.z_range = {};
			desc.w_range = Range{static_cast<int>(w_.size()), static_cast<int>(quad.weights.size())};

			x_.insert(x_.end(), col_x.begin(), col_x.end());
			w_.insert(w_.end(), quad.weights.begin(), quad.weights.end());
		}
		// dim == 2, z is empty.
		else if (desc.dim == 2)
		{
			auto col_x = quad.points.col(0);
			auto col_y = quad.points.col(1);
			desc.x_range = Range{static_cast<int>(x_.size()), static_cast<int>(col_x.size())};
			desc.y_range = Range{static_cast<int>(y_.size()), static_cast<int>(col_y.size())};
			desc.z_range = {};
			desc.w_range = Range{static_cast<int>(w_.size()), static_cast<int>(quad.weights.size())};

			x_.insert(x_.end(), col_x.begin(), col_x.end());
			y_.insert(y_.end(), col_y.begin(), col_y.end());
			w_.insert(w_.end(), quad.weights.begin(), quad.weights.end());
		}
		else if (desc.dim == 3)
		{
			auto col_x = quad.points.col(0);
			auto col_y = quad.points.col(1);
			auto col_z = quad.points.col(2);
			desc.x_range = Range{static_cast<int>(x_.size()), static_cast<int>(col_x.size())};
			desc.y_range = Range{static_cast<int>(y_.size()), static_cast<int>(col_y.size())};
			desc.z_range = Range{static_cast<int>(z_.size()), static_cast<int>(col_z.size())};
			desc.w_range = Range{static_cast<int>(w_.size()), static_cast<int>(quad.weights.size())};

			x_.insert(x_.end(), col_x.begin(), col_x.end());
			y_.insert(y_.end(), col_y.begin(), col_y.end());
			z_.insert(z_.end(), col_z.begin(), col_z.end());
			w_.insert(w_.end(), quad.weights.begin(), quad.weights.end());
		}
		else
		{
			assert(false && "Invalid dimension");
		}

		return desc;
	}

#ifdef POLYFEM_WITH_CUDA
	QuadratureStoreView QuadratureStore::device_view(CudaExecutionPolicy policy)
	{
		auto &p = policy;
		if (need_host_device_sync_)
		{
			d_x_ = cuda::make_buffer<double>(p.stream, p.mr, x_.size(), cuda::no_init);
			d_y_ = cuda::make_buffer<double>(p.stream, p.mr, y_.size(), cuda::no_init);
			d_z_ = cuda::make_buffer<double>(p.stream, p.mr, z_.size(), cuda::no_init);
			d_w_ = cuda::make_buffer<double>(p.stream, p.mr, w_.size(), cuda::no_init);
			cuda::copy_bytes(p.stream, x_, *d_x_);
			cuda::copy_bytes(p.stream, y_, *d_y_);
			cuda::copy_bytes(p.stream, z_, *d_z_);
			cuda::copy_bytes(p.stream, w_, *d_w_);
			need_host_device_sync_ = false;

			p.stream.sync();
		}
		return QuadratureStoreView{*d_x_, *d_y_, *d_z_, *d_w_};
	}

	void QuadratureStore::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_x_ = {};
		d_y_ = {};
		d_z_ = {};
		d_w_ = {};
	}
#endif

} // namespace polyfem::quadrature
