#include <polyfem/assembler/AssemblyEssentials.hpp>

#include <polyfem/basis/BasisStore.hpp>
#include <polyfem/basis/EvalBasis.hpp>
#include <polyfem/quadrature/QuadratureStore.hpp>
#include <polyfem/assembler/DofMappingStore.hpp>
#include <polyfem/assembler/AssemblyValsCache.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <Eigen/Core>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::assembler
{
	namespace
	{
		Eigen::MatrixXd span_components_to_points(
			const Span<const double> x,
			const Span<const double> y,
			const Span<const double> z,
			const int dim)
		{
			const int n = static_cast<int>(x.size());
			Eigen::MatrixXd pts(n, dim);
			for (int i = 0; i < n; ++i)
			{
				pts(i, 0) = x[i];
				if (dim > 1)
					pts(i, 1) = y[i];
				if (dim > 2)
					pts(i, 2) = z[i];
			}
			return pts;
		}

		void matrix_to_component_spans(
			const Eigen::MatrixXd &uv,
			std::vector<double> &x,
			std::vector<double> &y,
			std::vector<double> &z)
		{
			x.resize(uv.rows());
			y.resize(uv.cols() > 1 ? uv.rows() : 0);
			z.resize(uv.cols() > 2 ? uv.rows() : 0);
			for (int i = 0; i < uv.rows(); ++i)
			{
				x[i] = uv(i, 0);
				if (uv.cols() > 1)
					y[i] = uv(i, 1);
				if (uv.cols() > 2)
					z[i] = uv(i, 2);
			}
		}

		basis::Basis make_legacy_basis(
			const int element_id,
			const int local_basis_id,
			const ElementDesc &desc,
			const basis::BasisStoreView basis_store,
			const DofMappingStoreView dof_mapping_store)
		{
			const int dim = desc.basis_desc.dim;
			const int mapping_id = desc.dof_mapping_range.offset + local_basis_id;
			std::vector<basis::Local2Global> mapping = dof_mapping_store.get_local_to_global(mapping_id, dim);
			assert(!mapping.empty());

			basis::Basis basis;
			basis.init(desc.basis_desc.order, mapping.front().index, local_basis_id, mapping.front().node);
			basis.global() = std::move(mapping);

			basis.set_basis([element_id, local_basis_id, desc, basis_store](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
				assert(uv.cols() == desc.basis_desc.dim);
				std::vector<double> x, y, z;
				matrix_to_component_spans(uv, x, y, z);
				val.resize(uv.rows(), 1);
				basis::basis_values_single(
					local_basis_id,
					desc.basis_desc,
					basis_store,
					x,
					y,
					z,
					Span<double>(val.data(), val.size()));
			});

			basis.set_grad([element_id, local_basis_id, desc, basis_store](const Eigen::MatrixXd &uv, Eigen::MatrixXd &grad) {
				assert(uv.cols() == desc.basis_desc.dim);
				std::vector<double> x, y, z;
				matrix_to_component_spans(uv, x, y, z);
				std::vector<double> gx(uv.rows());
				std::vector<double> gy(desc.basis_desc.dim > 1 ? uv.rows() : 0);
				std::vector<double> gz(desc.basis_desc.dim > 2 ? uv.rows() : 0);
				basis::basis_gradients_single(
					local_basis_id,
					desc.basis_desc,
					basis_store,
					x,
					y,
					z,
					gx,
					gy,
					gz);

				grad.resize(uv.rows(), desc.basis_desc.dim);
				for (int i = 0; i < uv.rows(); ++i)
				{
					grad(i, 0) = gx[i];
					if (desc.basis_desc.dim > 1)
						grad(i, 1) = gy[i];
					if (desc.basis_desc.dim > 2)
						grad(i, 2) = gz[i];
				}
			});

			return basis;
		}
	} // namespace

	AssemblyEssentialsView AssemblyEssentials::view() const
	{
		return AssemblyEssentialsView{
			element_desc,
			quadrature_store.view(),
			mass_quadrature_store.view(),
			basis_store.view(),
			dof_mapping_store.view(),
			legacy_local_nodes_from_primitive};
	}

	const std::vector<basis::ElementBases> &AssemblyEssentials::legacy_bases() const
	{
		return *legacy_bases_ptr();
	}

	std::shared_ptr<std::vector<basis::ElementBases>> AssemblyEssentials::legacy_bases_ptr() const
	{
		if (legacy_bases_)
			return legacy_bases_;

		legacy_bases_ = std::make_shared<std::vector<basis::ElementBases>>();
		legacy_bases_->resize(element_desc.size());

		const auto basis_store_view = basis_store.view();
		const auto dof_mapping_store_view = dof_mapping_store.view();
		const auto quadrature_store_view = quadrature_store.view();
		const auto mass_quadrature_store_view = mass_quadrature_store.view();

		for (int e = 0; e < int(element_desc.size()); ++e)
		{
			const ElementDesc &desc = element_desc[e];
			basis::ElementBases &legacy = (*legacy_bases_)[e];
			legacy.has_parameterization = desc.basis_desc.is_parametric;

			const quadrature::Quadrature quadrature = quadrature_store_view.get_quadrature(desc.quadrature_desc);
			legacy.set_quadrature([quadrature](quadrature::Quadrature &out) { out = quadrature; });

			const quadrature::Quadrature mass_quadrature = mass_quadrature_store_view.get_quadrature(desc.mass_quadrature_desc);
			legacy.set_mass_quadrature([mass_quadrature](quadrature::Quadrature &out) { out = mass_quadrature; });

			if (e < int(legacy_local_nodes_from_primitive.size()) && legacy_local_nodes_from_primitive[e])
				legacy.set_local_node_from_primitive_func(legacy_local_nodes_from_primitive[e]);

			const int n_bases = desc.basis_desc.basis_num;
			legacy.bases.reserve(n_bases);
			for (int local_basis_id = 0; local_basis_id < n_bases; ++local_basis_id)
			{
				legacy.bases.push_back(make_legacy_basis(
					e,
					local_basis_id,
					desc,
					basis_store_view,
					dof_mapping_store_view));
			}
		}

		return legacy_bases_;
	}

	AssemblyValsCache AssemblyEssentials::legacy_assembly_vals_cache(
		const AssemblyEssentials &geom_bases,
		const bool is_volume,
		const bool is_mass) const
	{
		AssemblyValsCache cache;
		cache.init(is_volume, legacy_bases(), geom_bases.legacy_bases(), is_mass);
		return cache;
	}

	AssemblyValsCache AssemblyEssentials::legacy_empty_assembly_vals_cache(const bool is_mass)
	{
		AssemblyValsCache cache;
		cache.init_empty(is_mass);
		return cache;
	}

#ifdef POLYFEM_WITH_CUDA
	AssemblyEssentialsView AssemblyEssentials::device_view(CudaExecutionPolicy policy)
	{
		auto &p = policy;
		if (need_host_device_sync_)
		{
			d_element_desc_ = cuda::make_buffer<ElementDesc>(p.stream, p.mr, element_desc.size(), cuda::no_init);
			cuda::copy_bytes(p.stream, element_desc, *d_element_desc_);
			need_host_device_sync_ = false;
			p.stream.sync();
		}

		return AssemblyEssentialsView{
			*d_element_desc_,
			quadrature_store.device_view(p),
			mass_quadrature_store.device_view(p),
			basis_store.device_view(p),
			dof_mapping_store.device_view(p),
			{}};
	}

	void AssemblyEssentials::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_element_desc_ = {};
		quadrature_store.clear_device_storage();
		mass_quadrature_store.clear_device_storage();
		basis_store.clear_device_storage();
		dof_mapping_store.clear_device_storage();
	}
#endif

} // namespace polyfem::assembler
