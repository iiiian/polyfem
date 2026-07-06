#pragma once

#include <polyfem/basis/BasisStore.hpp>
#include <polyfem/quadrature/QuadratureStore.hpp>
#include <polyfem/assembler/DofMappingStore.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>
#include <Eigen/Core>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#endif

namespace polyfem::assembler
{

	struct ElementDesc
	{
		quadrature::QuadratureDesc quadrature_desc;
		quadrature::QuadratureDesc mass_quadrature_desc;
		basis::BasisDesc basis_desc;
		Range dof_mapping_range;
	};

	using LocalNodeFromPrimitiveFunc = std::function<Eigen::VectorXi(const int local_index, const mesh::Mesh &mesh)>;

	struct ElementBasesView
	{
		Span<const ElementDesc> element_desc;
		quadrature::QuadratureStoreView quadrature_store;
		quadrature::QuadratureStoreView mass_quadrature_store;
		basis::BasisStoreView basis_store;
		DofMappingStoreView dof_mapping_store;
		[[deprecated]] Span<const LocalNodeFromPrimitiveFunc> legacy_local_nodes_from_primitive;
	};

	class ElementBases
	{
	public:
		std::vector<ElementDesc> element_desc;
		quadrature::QuadratureStore quadrature_store;
		quadrature::QuadratureStore mass_quadrature_store;
		basis::BasisStore basis_store;
		DofMappingStore dof_mapping_store;

		[[deprecated]] std::vector<LocalNodeFromPrimitiveFunc> legacy_local_nodes_from_primitive;

#ifdef POLYFEM_WITH_CUDA
		bool need_host_device_sync_ = true;
#endif

	private:
#ifdef POLYFEM_WITH_CUDA
		DeviceBuf<ElementDesc> d_element_desc_;
#endif

	public:
		ElementBasesView view() const;

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device memory. Lazily sync data.
		ElementBasesView device_view(CudaExecutionPolicy policy = {});

		/// Release device storage.
		void clear_device_storage();
#endif
	};

} // namespace polyfem::assembler
