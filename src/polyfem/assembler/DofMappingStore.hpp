#pragma once

#include <polyfem/basis/Local2Global.hpp>
#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#endif

namespace polyfem::assembler
{

	struct DofMappingDesc
	{
		Range id_and_weight_range;
		Range node_position_range;
	};

	struct DofMappingStoreView
	{
		Span<const DofMappingDesc> mapping_desc;
		Span<const int> node_ids;
		Span<const double> weights;
		Span<const double> node_positions;

		POLYFEM_BOTH Span<const int> get_node_ids(int mapping_id) const
		{
			auto &desc = mapping_desc[mapping_id];
			return slice_by_range(node_ids, desc.id_and_weight_range);
		}
		POLYFEM_BOTH Span<const double> get_weights(int mapping_id) const
		{
			auto &desc = mapping_desc[mapping_id];
			return slice_by_range(weights, desc.id_and_weight_range);
		}
		POLYFEM_BOTH Span<const double> get_positions(int mapping_id) const
		{
			auto &desc = mapping_desc[mapping_id];
			return slice_by_range(node_positions, desc.node_position_range);
		}

		std::vector<basis::Local2Global> get_local_to_global(int mapping_id, int dim) const;
	};

	class DofMappingStore
	{
	private:
		std::vector<DofMappingDesc> mapping_desc_;
		std::vector<int> node_ids_;
		std::vector<double> weights_;
		std::vector<double> node_positions_;

#ifdef POLYFEM_WITH_CUDA
		bool need_host_device_sync_ = true;
		DeviceBuf<DofMappingDesc> d_mapping_desc_;
		DeviceBuf<int> d_node_ids_;
		DeviceBuf<double> d_weights_;
		DeviceBuf<double> d_node_positions_;
#endif

	public:
		DofMappingStoreView view() const;

		int append(Span<const int> node_ids, Span<const double> weights, Span<const double> node_positions);

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device memory. Lazily sync data.
		DofMappingStoreView device_view(CudaExecutionPolicy policy = {});

		/// Release device storage.
		void clear_device_storage();
#endif
	};

} // namespace polyfem::assembler
