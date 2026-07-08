#pragma once

#include <polyfem/materials/MaterialExprRegistry.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/ComputeAssemblyCache.hpp>
#include <polyfem/assembler/ElementBases.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Span.hpp>

#include <ipc/utils/eigen_ext.hpp>

#include <Eigen/Core>

#include <cassert>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace polyfem::assembler
{
	namespace host_detail
	{
		using RMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

		inline void atomic_add(double &target, double value)
		{
#if defined(_MSC_VER)
			static_assert(sizeof(double) == sizeof(__int64), "Expected double and __int64 to have the same size.");
			assert(reinterpret_cast<std::uintptr_t>(&target) % alignof(__int64) == 0);

			auto *target_bits = reinterpret_cast<volatile __int64 *>(&target);
			__int64 old_bits = _InterlockedCompareExchange64(target_bits, 0, 0);
			while (true)
			{
				double old_value;
				std::memcpy(&old_value, &old_bits, sizeof(old_value));

				double new_value = old_value + value;
				__int64 new_bits;
				std::memcpy(&new_bits, &new_value, sizeof(new_bits));

				__int64 observed = _InterlockedCompareExchange64(target_bits, new_bits, old_bits);
				if (observed == old_bits)
					break;

				old_bits = observed;
			}
#else
			double old_value;
			__atomic_load(&target, &old_value, __ATOMIC_RELAXED);
			while (true)
			{
				double new_value = old_value + value;
				if (__atomic_compare_exchange(
						&target,
						&old_value,
						&new_value,
						true,
						__ATOMIC_RELAXED,
						__ATOMIC_RELAXED))
				{
					break;
				}
			}
#endif
		}

		inline void assert_cache_compatible(int element_num, AssemblyCacheView cache)
		{
			if (cache.desc.empty())
			{
				return;
			}
			if (cache.desc.size() != element_num)
			{
				throw std::runtime_error("CPU assembler cache descriptor count does not match element count.");
			}
		}

		template <int dim>
		ElementAssemblyCacheView compute_element_cache(
			const ElementBasesView &bases,
			const ElementBasesView &geom_bases,
			int element_id,
			AssemblyTempStorage &temp,
			AssemblyCache &temp_cache)
		{
			temp_cache.clear();
			compute_assembly_cache_single<dim>(bases, geom_bases, element_id, false, temp);
			int cache_element_id = temp_cache.append(false, temp);
			assert(cache_element_id == 0);
			return temp_cache.view().slice(cache_element_id);
		}

		inline ElementAssemblyCacheView element_cache_view(
			const ElementBasesView &bases,
			const ElementBasesView &geom_bases,
			AssemblyCacheView cache,
			int element_id,
			AssemblyTempStorage &temp,
			AssemblyCache &temp_cache)
		{
			if (!cache.desc.empty() && !cache.desc[element_id].is_empty)
			{
				return cache.slice(element_id);
			}

			int dim = bases.element_desc[element_id].basis_desc.dim;
			switch (dim)
			{
			case 1:
				return compute_element_cache<1>(bases, geom_bases, element_id, temp, temp_cache);
			case 2:
				return compute_element_cache<2>(bases, geom_bases, element_id, temp, temp_cache);
			case 3:
				return compute_element_cache<3>(bases, geom_bases, element_id, temp, temp_cache);
			default:
				throw std::runtime_error("Unsupported element dimension in CPU assembly.");
			}
		}

		template <typename Material, int dim>
		Material eval_material(
			const material::MaterialExprRegistry &material_registry,
			ElementAssemblyCacheView cache,
			int element_id,
			int quad_id,
			double time)
		{
			const auto *material_expr = material_registry.template get<typename Material::ExprType>(element_id);
			if (material_expr == nullptr)
			{
				throw std::runtime_error("Material missing in CPU assembly.");
			}

			double x = cache.get_physical_x(quad_id);
			double y = 0.0;
			double z = 0.0;
			if constexpr (dim > 1)
			{
				y = cache.get_physical_y(quad_id);
			}
			if constexpr (dim > 2)
			{
				z = cache.get_physical_z(quad_id);
			}
			return material_expr->eval_expr(x, y, z, time, element_id);
		}

		template <typename ScalarKernel>
		double assemble_element_scalar(
			const ElementBasesView &bases,
			ElementAssemblyCacheView cache,
			const material::MaterialExprRegistry &material_registry,
			Span<const double> unknown,
			int element_id,
			double time)
		{
			using Material = typename ScalarKernel::Material;
			constexpr int DIM = ScalarKernel::DIM;

			int quad_num = cache.quad_num();
			double scalar = 0.0;
			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				const Material material = eval_material<Material, DIM>(material_registry, cache, element_id, quad_id, time);
				scalar += ScalarKernel::eval_scalar(element_id, quad_id, bases, cache, material, unknown)
						  * cache.get_weighted_measure(quad_id);
			}
			return scalar;
		}

		template <typename VectorKernel>
		void assemble_element_vector(
			const ElementBasesView &bases,
			ElementAssemblyCacheView cache,
			const material::MaterialExprRegistry &material_registry,
			Span<const double> unknown,
			int element_id,
			double time,
			Span<double> local_vector)
		{
			using Material = typename VectorKernel::Material;
			constexpr int VALUE_DIM = VectorKernel::VALUE_DIM;
			constexpr int DIM = VectorKernel::DIM;
			using Vec = Eigen::Vector<double, VALUE_DIM>;

			const auto &elem_desc = bases.element_desc[element_id];
			int basis_num = elem_desc.basis_desc.basis_num;
			int quad_num = cache.quad_num();
			assert(local_vector.size() == basis_num * VALUE_DIM);

			for (int basis_id = 0; basis_id < basis_num; ++basis_id)
			{
				Vec grad_i = Vec::Zero();
				for (int quad_id = 0; quad_id < quad_num; ++quad_id)
				{
					const Material material = eval_material<Material, DIM>(material_registry, cache, element_id, quad_id, time);
					Vec local_grad_i = Vec::Zero();
					Span<double> local_grad_i_span(local_grad_i.data(), local_grad_i.size());
					VectorKernel::eval_vector(
						element_id,
						quad_id,
						basis_id,
						bases,
						cache,
						material,
						unknown,
						local_grad_i_span);
					grad_i += local_grad_i * cache.get_weighted_measure(quad_id);
				}

				for (int d = 0; d < VALUE_DIM; ++d)
				{
					local_vector[basis_id * VALUE_DIM + d] = grad_i(d);
				}
			}
		}

		template <typename MatrixKernel>
		void assemble_element_matrix(
			const ElementBasesView &bases,
			ElementAssemblyCacheView cache,
			const material::MaterialExprRegistry &material_registry,
			Span<const double> unknown,
			int element_id,
			double time,
			RMatrixXd &local_matrix)
		{
			using Material = typename MatrixKernel::Material;
			constexpr int VALUE_DIM = MatrixKernel::VALUE_DIM;
			constexpr int DIM = MatrixKernel::DIM;
			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;

			const auto &elem_desc = bases.element_desc[element_id];
			int basis_num = elem_desc.basis_desc.basis_num;
			int local_dof_num = basis_num * VALUE_DIM;
			int quad_num = cache.quad_num();
			assert(local_matrix.rows() == local_dof_num);
			assert(local_matrix.cols() == local_dof_num);

			for (int bi = 0; bi < basis_num; ++bi)
			{
				for (int bj = bi; bj < basis_num; ++bj)
				{
					Mat hess_ij = Mat::Zero();
					for (int quad_id = 0; quad_id < quad_num; ++quad_id)
					{
						const Material material = eval_material<Material, DIM>(material_registry, cache, element_id, quad_id, time);
						Mat local_hess_ij = Mat::Zero();
						Span<double> hess_ij_span(local_hess_ij.data(), local_hess_ij.size());
						MatrixKernel::eval_matrix(
							element_id,
							quad_id,
							bi,
							bj,
							bases,
							cache,
							material,
							unknown,
							hess_ij_span);
						hess_ij += local_hess_ij * cache.get_weighted_measure(quad_id);
					}

					local_matrix.block<VALUE_DIM, VALUE_DIM>(bi * VALUE_DIM, bj * VALUE_DIM) = hess_ij;

					if (bj > bi)
					{
						local_matrix.block<VALUE_DIM, VALUE_DIM>(bj * VALUE_DIM, bi * VALUE_DIM) = hess_ij.transpose();
					}
				}
			}
		}

		inline void scatter_element_vector(
			const ElementDesc &element_desc,
			DofMappingStoreView mapping,
			Span<const double> local_vector,
			int basis_num,
			int value_dim,
			Span<double> global_vector)
		{
			assert(local_vector.size() == basis_num * value_dim);
			for (int basis_id = 0; basis_id < basis_num; ++basis_id)
			{
				int mapping_id = element_desc.dof_mapping_range.offset + basis_id;
				auto node_ids = mapping.get_node_ids(mapping_id);
				auto node_weights = mapping.get_weights(mapping_id);
				assert(node_ids.size() == node_weights.size());

				for (int d = 0; d < value_dim; ++d)
				{
					double local_value = local_vector[basis_id * value_dim + d];
					if (local_value == 0.0)
					{
						continue;
					}

					for (int node_id = 0; node_id < node_ids.size(); ++node_id)
					{
						int offset = node_ids[node_id] * value_dim + d;
						assert(offset >= 0 && offset < global_vector.size());
						atomic_add(global_vector[offset], node_weights[node_id] * local_value);
					}
				}
			}
		}

		inline void scatter_element_matrix(
			const ElementDesc &element_desc,
			DofMappingStoreView mapping,
			const RMatrixXd &local_matrix,
			int basis_num,
			int value_dim,
			BSRMatrixMutableView global_matrix)
		{
			assert(value_dim == global_matrix.block_dim);
			int local_dof_num = basis_num * value_dim;
			assert(local_matrix.rows() == local_dof_num);
			assert(local_matrix.cols() == local_dof_num);

			for (int bi = 0; bi < basis_num; ++bi)
			{
				int row_mapping_id = element_desc.dof_mapping_range.offset + bi;
				auto row_node_ids = mapping.get_node_ids(row_mapping_id);
				auto row_weights = mapping.get_weights(row_mapping_id);
				assert(row_node_ids.size() == row_weights.size());

				for (int bj = 0; bj < basis_num; ++bj)
				{
					auto local_block = local_matrix.block(
						bi * value_dim,
						bj * value_dim,
						value_dim,
						value_dim);
					if (local_block.isZero())
					{
						continue;
					}

					int col_mapping_id = element_desc.dof_mapping_range.offset + bj;
					auto col_node_ids = mapping.get_node_ids(col_mapping_id);
					auto col_weights = mapping.get_weights(col_mapping_id);
					assert(col_node_ids.size() == col_weights.size());

					for (int row_node_id = 0; row_node_id < row_node_ids.size(); ++row_node_id)
					{
						for (int col_node_id = 0; col_node_id < col_node_ids.size(); ++col_node_id)
						{
							double *block_ptr = global_matrix.get_block(row_node_ids[row_node_id], col_node_ids[col_node_id]);
							assert(block_ptr);

							double weight = row_weights[row_node_id] * col_weights[col_node_id];
							for (int r = 0; r < value_dim; ++r)
							{
								double *dst = block_ptr + r * value_dim;
								for (int c = 0; c < value_dim; ++c)
								{
									double value = local_block(r, c);
									if (value != 0.0)
									{
										atomic_add(dst[c], weight * value);
									}
								}
							}
						}
					}
				}
			}
		}
	} // namespace host_detail

	template <typename ScalarKernel>
	double assemble_scalar(
		const ElementBases &bases,
		const ElementBases &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		double time = 0.0)
	{
		const ElementBasesView bases_view = bases.view();
		const ElementBasesView geom_bases_view = geom_bases.view();
		const AssemblyCacheView cache_view = cache.view();
		int element_num = static_cast<int>(bases.element_desc.size());
		host_detail::assert_cache_compatible(element_num, cache_view);

		double scalar_out = 0.0;
		utils::maybe_parallel_for(element_num, [&](int start, int end, int) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;
			double local_scalar = 0.0;
			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				ElementAssemblyCacheView elem_cache = host_detail::element_cache_view(
					bases_view,
					geom_bases_view,
					cache_view,
					elem_id,
					temp,
					temp_cache);
				local_scalar += host_detail::assemble_element_scalar<ScalarKernel>(
					bases_view,
					elem_cache,
					material_registry,
					unknown,
					elem_id,
					time);
			}
			if (local_scalar != 0.0)
			{
				host_detail::atomic_add(scalar_out, local_scalar);
			}
		});

		return scalar_out;
	}

	template <typename ScalarKernel>
	void assemble_scalar_per_element(
		const ElementBases &bases,
		const ElementBases &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> scalar_out,
		double time = 0.0)
	{
		const ElementBasesView bases_view = bases.view();
		const ElementBasesView geom_bases_view = geom_bases.view();
		const AssemblyCacheView cache_view = cache.view();
		int element_num = static_cast<int>(bases.element_desc.size());
		assert(scalar_out.size() == element_num);
		host_detail::assert_cache_compatible(element_num, cache_view);

		utils::maybe_parallel_for(element_num, [&](int start, int end, int) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;
			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				ElementAssemblyCacheView elem_cache = host_detail::element_cache_view(
					bases_view,
					geom_bases_view,
					cache_view,
					elem_id,
					temp,
					temp_cache);
				scalar_out[elem_id] += host_detail::assemble_element_scalar<ScalarKernel>(
					bases_view,
					elem_cache,
					material_registry,
					unknown,
					elem_id,
					time);
			}
		});
	}

	template <typename VectorKernel>
	void assemble_vector(
		const ElementBases &bases,
		const ElementBases &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vector_out,
		double time = 0.0)
	{
		constexpr int VALUE_DIM = VectorKernel::VALUE_DIM;

		const ElementBasesView bases_view = bases.view();
		const ElementBasesView geom_bases_view = geom_bases.view();
		const AssemblyCacheView cache_view = cache.view();
		int element_num = static_cast<int>(bases.element_desc.size());
		host_detail::assert_cache_compatible(element_num, cache_view);

		utils::maybe_parallel_for(element_num, [&](int start, int end, int) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;
			std::vector<double> local_vector_storage;
			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				const ElementDesc &elem_desc = bases_view.element_desc[elem_id];
				int basis_num = elem_desc.basis_desc.basis_num;
				local_vector_storage.resize(basis_num * VALUE_DIM);
				Span<double> local_vector(local_vector_storage.data(), local_vector_storage.size());
				ElementAssemblyCacheView elem_cache = host_detail::element_cache_view(
					bases_view,
					geom_bases_view,
					cache_view,
					elem_id,
					temp,
					temp_cache);
				host_detail::assemble_element_vector<VectorKernel>(
					bases_view,
					elem_cache,
					material_registry,
					unknown,
					elem_id,
					time,
					local_vector);

				host_detail::scatter_element_vector(
					elem_desc,
					bases_view.dof_mapping_store,
					Span<const double>(local_vector.data(), local_vector.size()),
					basis_num,
					VALUE_DIM,
					vector_out);
			}
		});
	}

	template <typename MatrixKernel>
	void assemble_matrix(
		const ElementBases &bases,
		const ElementBases &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		BSRMatrixMutableView matrix_out,
		bool project_to_psd = false,
		double time = 0.0)
	{
		constexpr int VALUE_DIM = MatrixKernel::VALUE_DIM;
		assert(matrix_out.block_dim == VALUE_DIM);

		const ElementBasesView bases_view = bases.view();
		const ElementBasesView geom_bases_view = geom_bases.view();
		const AssemblyCacheView cache_view = cache.view();
		int element_num = static_cast<int>(bases.element_desc.size());
		host_detail::assert_cache_compatible(element_num, cache_view);

		utils::maybe_parallel_for(element_num, [&](int start, int end, int) {
			AssemblyTempStorage temp;
			AssemblyCache temp_cache;
			host_detail::RMatrixXd local_matrix;
			for (int elem_id = start; elem_id < end; ++elem_id)
			{
				const ElementDesc &elem_desc = bases_view.element_desc[elem_id];
				int basis_num = elem_desc.basis_desc.basis_num;
				int local_dof_num = basis_num * VALUE_DIM;
				local_matrix.resize(local_dof_num, local_dof_num);
				ElementAssemblyCacheView elem_cache = host_detail::element_cache_view(
					bases_view,
					geom_bases_view,
					cache_view,
					elem_id,
					temp,
					temp_cache);
				host_detail::assemble_element_matrix<MatrixKernel>(
					bases_view,
					elem_cache,
					material_registry,
					unknown,
					elem_id,
					time,
					local_matrix);

				if (project_to_psd)
				{
					local_matrix = ipc::project_to_psd(local_matrix);
				}

				host_detail::scatter_element_matrix(
					elem_desc,
					bases_view.dof_mapping_store,
					local_matrix,
					basis_num,
					VALUE_DIM,
					matrix_out);
			}
		});
	}
} // namespace polyfem::assembler
