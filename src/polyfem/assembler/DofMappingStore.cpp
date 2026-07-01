#include <polyfem/assembler/DofMappingStore.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::assembler
{

	DofMappingStoreView DofMappingStore::view() const
	{
		return DofMappingStoreView{mapping_desc_, node_ids_, weights_, node_positions_};
	}

	int DofMappingStore::append(Span<const int> node_ids, Span<const double> weights, Span<const double> node_positions)
	{
		assert(node_ids.size() != 0 && node_ids.size() == weights.size());
		assert(node_positions.size() != 0);

		DofMappingDesc desc;
		desc.id_and_weight_range = Range{static_cast<int>(this->node_ids_.size()),
										 static_cast<int>(node_ids.size())};
		desc.node_position_range = Range{static_cast<int>(this->node_positions_.size()),
										 static_cast<int>(node_positions.size())};

		this->node_ids_.insert(this->node_ids_.end(), node_ids.begin(), node_ids.end());
		this->weights_.insert(this->weights_.end(), weights.begin(), weights.end());
		this->node_positions_.insert(this->node_positions_.end(), node_positions.begin(), node_positions.end());
		this->mapping_desc_.push_back(desc);

#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif
		return this->mapping_desc_.size() - 1;
	}

#ifdef POLYFEM_WITH_CUDA
	DofMappingStoreView DofMappingStore::device_view(CudaExecutionPolicy policy)
	{

		auto &p = policy;
		if (need_host_device_sync_)
		{
			d_mapping_desc_ = cuda::make_buffer<DofMappingDesc>(p.stream, p.mr, mapping_desc_.size(), cuda::no_init);
			d_node_ids_ = cuda::make_buffer<int>(p.stream, p.mr, node_ids_.size(), cuda::no_init);
			d_weights_ = cuda::make_buffer<double>(p.stream, p.mr, weights_.size(), cuda::no_init);
			d_node_positions_ = cuda::make_buffer<double>(p.stream, p.mr, node_positions_.size(), cuda::no_init);
			cuda::copy_bytes(p.stream, mapping_desc_, *d_mapping_desc_);
			cuda::copy_bytes(p.stream, node_ids_, *d_node_ids_);
			cuda::copy_bytes(p.stream, weights_, *d_weights_);
			cuda::copy_bytes(p.stream, node_positions_, *d_node_positions_);
			need_host_device_sync_ = false;

			p.stream.sync();
		}
		return DofMappingStoreView{*d_mapping_desc_, *d_node_ids_, *d_weights_, *d_node_positions_};
	}

	void DofMappingStore::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_mapping_desc_ = {};
		d_node_ids_ = {};
		d_weights_ = {};
		d_node_positions_ = {};
	}
#endif

} // namespace polyfem::assembler
