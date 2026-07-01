#include "polyfem/basis/Basis.hpp"
#include "polyfem/basis/ElementBases.hpp"
#include <polyfem/basis/BasisStore.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::basis
{
	Range BasisStore::append_rational_weights(Span<const double> weights)
	{
		Range r{static_cast<int>(rational_weights_.size()), static_cast<int>(weights.size())};
		rational_weights_.insert(rational_weights_.end(), weights.begin(), weights.end());
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif
		return r;
	}

	int BasisStore::append_eval_callback(BasisEvalCallback callback)
	{
		int idx = eval_callbacks_.size();
		eval_callbacks_.push_back(std::move(callback));
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif
		return idx;
	}

	BasisStoreView BasisStore::view() const { return BasisStoreView{rational_weights_}; }

#ifdef POLYFEM_WITH_CUDA
	/// Return view on device memory. Lazily sync data.
	BasisStoreView BasisStore::device_view(CudaExecutionPolicy policy)
	{
		auto &p = policy;

		if (need_host_device_sync_)
		{
			d_rational_weights_ = cuda::make_buffer<double>(p.stream, p.mr, rational_weights_.size(), cuda::no_init);
			cuda::copy_bytes(p.stream, rational_weights_, *d_rational_weights_);
			need_host_device_sync_ = false;

			p.stream.sync();
		}
		return BasisStoreView{*d_rational_weights_, {}};
	}

	/// Release device storage.
	void BasisStore::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_rational_weights_ = {};
	}
#endif

} // namespace polyfem::basis
