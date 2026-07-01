#pragma once

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
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

		Span<const int> get_node_ids(int element_id, int basis_id) const
		{
			auto &desc = mapping_desc[element_id];
			return slice_by_range(node_ids, desc.id_and_weight_range);
		}
		Span<const double> get_weights(int element_id, int basis_id) const
		{
			auto &desc = mapping_desc[element_id];
			return slice_by_range(weights, desc.id_and_weight_range);
		}
		Span<const double> get_positions(int element_id, int basis_id) const
		{
			auto &desc = mapping_desc[element_id];
			return slice_by_range(node_positions, desc.node_position_range);
		}
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
		DBuf<DofMappingDesc> d_mapping_desc_;
		DBuf<int> d_node_ids_;
		DBuf<double> d_weights_;
		DBuf<double> d_node_positions_;
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
