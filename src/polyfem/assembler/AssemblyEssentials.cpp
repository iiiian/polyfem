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

		bool is_built_element(const ElementDesc &desc)
		{
			return desc.quadrature_desc.dim >= 1
				   && desc.quadrature_desc.dim <= 3
				   && desc.mass_quadrature_desc.dim >= 1
				   && desc.mass_quadrature_desc.dim <= 3
				   && desc.basis_desc.dim >= 1
				   && desc.basis_desc.dim <= 3
				   && desc.basis_desc.basis_num > 0
				   && desc.dof_mapping_range;
		}

#ifdef POLYFEM_WITH_CUDA
		std::vector<DeviceVectorAssemblyTask> build_device_vector_assembly_tasks(
			const std::vector<ElementDesc> &element_desc)
		{
			int task_num = 0;
			for (const ElementDesc &elem_desc : element_desc)
			{
				task_num += elem_desc.basis_desc.basis_num;
			}

			std::vector<DeviceVectorAssemblyTask> tasks;
			tasks.reserve(task_num);
			for (int elem_id = 0; elem_id < int(element_desc.size()); ++elem_id)
			{
				const int basis_num = element_desc[elem_id].basis_desc.basis_num;
				for (int bi = 0; bi < basis_num; ++bi)
				{
					tasks.push_back({elem_id, bi});
				}
			}

			return tasks;
		}

		std::vector<DeviceMatrixAssemblyTask> build_device_matrix_assembly_tasks(
			const std::vector<ElementDesc> &element_desc)
		{
			int task_num = 0;
			for (const ElementDesc &elem_desc : element_desc)
			{
				const int basis_num = elem_desc.basis_desc.basis_num;
				task_num += basis_num * (basis_num + 1) / 2;
			}

			std::vector<DeviceMatrixAssemblyTask> tasks;
			tasks.reserve(task_num);
			for (int elem_id = 0; elem_id < int(element_desc.size()); ++elem_id)
			{
				const int basis_num = element_desc[elem_id].basis_desc.basis_num;
				for (int bi = 0; bi < basis_num; ++bi)
				{
					for (int bj = bi; bj < basis_num; ++bj)
					{
						tasks.push_back({elem_id, bi, bj});
					}
				}
			}

			return tasks;
		}
#endif

		basis::BasisEvalCallback make_legacy_eval_callback(
			const basis::ElementBases &legacy_element,
			const int dim)
		{
			return [legacy_element, dim](
					   const Span<const double> x,
					   const Span<const double> y,
					   const Span<const double> z,
					   Span<double> values,
					   Span<double> grad_x,
					   Span<double> grad_y,
					   Span<double> grad_z) {
				const int n_points = static_cast<int>(x.size());
				const int n_bases = static_cast<int>(legacy_element.bases.size());
				assert(values.size() == n_bases * n_points);
				assert(grad_x.size() == n_bases * n_points);
				assert(dim < 2 || grad_y.size() == n_bases * n_points);
				assert(dim < 3 || grad_z.size() == n_bases * n_points);

				const Eigen::MatrixXd pts = span_components_to_points(x, y, z, dim);
				std::vector<AssemblyValues> basis_values;
				std::vector<AssemblyValues> basis_grads;
				legacy_element.evaluate_bases(pts, basis_values);
				legacy_element.evaluate_grads(pts, basis_grads);
				assert(int(basis_values.size()) == n_bases);
				assert(int(basis_grads.size()) == n_bases);

				for (int local_basis_id = 0; local_basis_id < n_bases; ++local_basis_id)
				{
					assert(basis_values[local_basis_id].val.size() == n_points);
					assert(basis_grads[local_basis_id].grad.rows() == n_points);
					assert(basis_grads[local_basis_id].grad.cols() >= dim);

					const int offset = local_basis_id * n_points;
					for (int q = 0; q < n_points; ++q)
					{
						values[offset + q] = basis_values[local_basis_id].val(q);
						grad_x[offset + q] = basis_grads[local_basis_id].grad(q, 0);
						if (dim > 1)
							grad_y[offset + q] = basis_grads[local_basis_id].grad(q, 1);
						if (dim > 2)
							grad_z[offset + q] = basis_grads[local_basis_id].grad(q, 2);
					}
				}
			};
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
			if (!is_built_element(desc))
				continue;

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

	void AssemblyEssentials::set_legacy_element(
		const int element_id,
		const basis::ElementBases &legacy_element,
		LocalNodeFromPrimitiveFunc local_node_from_primitive)
	{
		assert(element_id >= 0);
		assert(element_id < int(element_desc.size()));
		assert(!legacy_element.bases.empty());

		quadrature::Quadrature quadrature;
		legacy_element.compute_quadrature(quadrature);
		quadrature::Quadrature mass_quadrature;
		legacy_element.compute_mass_quadrature(mass_quadrature);
		assert(quadrature.size() > 0);
		assert(mass_quadrature.size() > 0);
		assert(quadrature.points.cols() == mass_quadrature.points.cols());
		const int dim = quadrature.points.cols();

		ElementDesc desc{};
		desc.quadrature_desc = quadrature_store.append(quadrature);
		desc.mass_quadrature_desc = mass_quadrature_store.append(mass_quadrature);

		auto &basis_desc = desc.basis_desc;
		basis_desc.element_kind = dim == 3 ? basis::ElementKind::Polyhedron : basis::ElementKind::Polygon;
		basis_desc.basis_family = basis::BasisFamily::Unknown;
		basis_desc.order = legacy_element.bases.front().order();
		basis_desc.orderq = basis_desc.order;
		basis_desc.dim = dim;
		basis_desc.basis_num = int(legacy_element.bases.size());
		basis_desc.eval_callback_id = basis_store.append_eval_callback(make_legacy_eval_callback(legacy_element, dim));
		basis_desc.is_parametric = legacy_element.has_parameterization;
		basis_desc.is_bernstein = false;

		int first_mapping_id = 0;
		for (int local_basis_id = 0; local_basis_id < int(legacy_element.bases.size()); ++local_basis_id)
		{
			const auto &mapping = legacy_element.bases[local_basis_id].global();
			assert(!mapping.empty());

			std::vector<int> node_ids;
			std::vector<double> weights;
			std::vector<double> node_positions;
			node_ids.reserve(mapping.size());
			weights.reserve(mapping.size());
			node_positions.reserve(mapping.size() * dim);

			for (const auto &entry : mapping)
			{
				assert(entry.node.size() >= dim);
				node_ids.push_back(entry.index);
				weights.push_back(entry.val);
				for (int d = 0; d < dim; ++d)
					node_positions.push_back(entry.node(d));
			}

			const int mapping_id = dof_mapping_store.append(node_ids, weights, node_positions);
			if (local_basis_id == 0)
				first_mapping_id = mapping_id;
		}
		desc.dof_mapping_range = Range{first_mapping_id, basis_desc.basis_num};

		element_desc[element_id] = desc;
		if (legacy_local_nodes_from_primitive.size() < element_desc.size())
			legacy_local_nodes_from_primitive.resize(element_desc.size());
		legacy_local_nodes_from_primitive[element_id] = std::move(local_node_from_primitive);

		if (legacy_bases_)
			(*legacy_bases_)[element_id] = legacy_element;

#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif
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
	AssemblyEssentialsView AssemblyEssentials::device_view(ExecutionPolicy policy) const
	{
		auto &p = policy;
		if (need_host_device_sync_)
		{
			d_element_desc_ = cuda::make_buffer<ElementDesc>(*p.stream, *p.mr, element_desc.size(), cuda::no_init);
			cuda::copy_bytes(*p.stream, element_desc, *d_element_desc_);
			need_host_device_sync_ = false;
			p.stream->sync();
		}

		return AssemblyEssentialsView{
			*d_element_desc_,
			quadrature_store.device_view(p),
			mass_quadrature_store.device_view(p),
			basis_store.device_view(p),
			dof_mapping_store.device_view(p),
			{}};
	}

	Span<const DeviceVectorAssemblyTask> AssemblyEssentials::device_vector_assembly_tasks(ExecutionPolicy policy) const
	{
		if (!d_vector_assembly_tasks_)
		{
			std::vector<DeviceVectorAssemblyTask> tasks = build_device_vector_assembly_tasks(element_desc);
			d_vector_assembly_tasks_ = cuda::make_buffer<DeviceVectorAssemblyTask>(
				*policy.stream,
				*policy.mr,
				tasks.size(),
				cuda::no_init);
			cuda::copy_bytes(*policy.stream, tasks, *d_vector_assembly_tasks_);
			policy.stream->sync();
		}

		return *d_vector_assembly_tasks_;
	}

	Span<const DeviceMatrixAssemblyTask> AssemblyEssentials::device_matrix_assembly_tasks(ExecutionPolicy policy) const
	{
		if (!d_matrix_assembly_tasks_)
		{
			std::vector<DeviceMatrixAssemblyTask> tasks = build_device_matrix_assembly_tasks(element_desc);
			d_matrix_assembly_tasks_ = cuda::make_buffer<DeviceMatrixAssemblyTask>(
				*policy.stream,
				*policy.mr,
				tasks.size(),
				cuda::no_init);
			cuda::copy_bytes(*policy.stream, tasks, *d_matrix_assembly_tasks_);
			policy.stream->sync();
		}

		return *d_matrix_assembly_tasks_;
	}

	void AssemblyEssentials::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_element_desc_ = {};
		d_vector_assembly_tasks_ = {};
		d_matrix_assembly_tasks_ = {};
		quadrature_store.clear_device_storage();
		mass_quadrature_store.clear_device_storage();
		basis_store.clear_device_storage();
		dof_mapping_store.clear_device_storage();
	}
#endif

} // namespace polyfem::assembler
