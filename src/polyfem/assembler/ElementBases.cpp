#include <polyfem/assembler/ElementBases.hpp>

#include <polyfem/basis/BasisStore.hpp>
#include <polyfem/quadrature/QuadratureStore.hpp>
#include <polyfem/assembler/DofMappingStore.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <Eigen/Core>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::assembler
{

	ElementBasesView ElementBases::view() const
	{
		return ElementBasesView{
			element_desc,
			quadrature.view(),
			mass_quadrature.view(),
			basis.view(),
			dof_mapping.view(),
			legacy_local_nodes_from_primitive};
	}

#ifdef POLYFEM_WITH_CUDA
	ElementBasesView ElementBases::device_view(CudaExecutionPolicy policy)
	{
		auto &p = policy;
		if (need_host_device_sync_)
		{
			d_element_desc_ = cuda::make_buffer<ElementDesc>(p.stream, p.mr, element_desc.size(), cuda::no_init);
			cuda::copy_bytes(p.stream, element_desc, *d_element_desc_);
			need_host_device_sync_ = false;
			p.stream.sync();
		}

		return ElementBasesView{
			*d_element_desc_,
			quadrature.device_view(p),
			mass_quadrature.device_view(p),
			basis.device_view(p),
			dof_mapping.device_view(p),
			{}};
	}

	void ElementBases::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_element_desc_ = {};
		quadrature.clear_device_storage();
		mass_quadrature.clear_device_storage();
		basis.clear_device_storage();
		dof_mapping.clear_device_storage();
	}
#endif

} // namespace polyfem::assembler
