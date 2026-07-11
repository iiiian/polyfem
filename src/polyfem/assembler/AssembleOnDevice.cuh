#pragma once

#include <polyfem/materials/MaterialExprRegistry.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/AssemblyEssentials.hpp>
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/BlockCSRMatrix.hpp>
#include <polyfem/utils/CUDAUtils.hpp>

#include <cuda/buffer>
#include <cuda/algorithm>
#include <cuda/warp>
#include <cuda/std/utility>
#include <cub/cub.cuh>
#include <Eigen/Core>

#include <cassert>
#include <cstddef>
#include <vector>

namespace polyfem::assembler
{
	namespace detail
	{
		template <int VALUE_DIM>
		__device__ void scatter_mat_ij(
			int elem_id,
			int basis_i,
			int basis_j,
			AssemblyEssentialsView bases,
			Span<const double> local_mat,
			BSRMatrixMutableView global_mat)
		{
			assert(local_mat.size() == VALUE_DIM * VALUE_DIM);

			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;
			auto mat = Eigen::Map<const Mat>(local_mat.data());

			auto &elem_desc = bases.element_desc[elem_id];
			auto &mappings = bases.dof_mapping_store;
			int row_mapping_id = elem_desc.dof_mapping_range.offset + basis_i;
			int col_mapping_id = elem_desc.dof_mapping_range.offset + basis_j;

			auto row_node_ids = mappings.get_node_ids(row_mapping_id);
			auto row_node_weights = mappings.get_weights(row_mapping_id);
			auto col_node_ids = mappings.get_node_ids(col_mapping_id);
			auto col_node_weights = mappings.get_weights(col_mapping_id);
			assert(row_node_ids.size() == row_node_weights.size());
			assert(col_node_ids.size() == col_node_weights.size());

			for (int i = 0; i < row_node_ids.size(); ++i)
			{
				for (int j = 0; j < col_node_ids.size(); ++j)
				{
					double weight = row_node_weights[i] * col_node_weights[j];
					for (int r = 0; r < VALUE_DIM; ++r)
					{
						for (int c = 0; c < VALUE_DIM; ++c)
						{
							double value = mat(r, c);
							if (value == 0.0)
							{
								continue;
							}

							int global_row = row_node_ids[i] * VALUE_DIM + r;
							int global_col = col_node_ids[j] * VALUE_DIM + c;
							double *dst = global_mat.get_entry(global_row, global_col);
							assert(dst);
							atomicAdd(dst, weight * value);
						}
					}
				}
			}
		}

		template <int value_dim>
		__device__ void scatter_vec_i(
			int elem_id,
			int basis_i,
			AssemblyEssentialsView bases,
			Span<const double> local_vec,
			Span<double> global_vec)
		{
			assert(local_vec.size() == value_dim);

			auto &elem_desc = bases.element_desc[elem_id];
			auto &mappings = bases.dof_mapping_store;
			int mapping_id = elem_desc.dof_mapping_range.offset + basis_i;

			auto node_ids = mappings.get_node_ids(mapping_id);
			auto node_weights = mappings.get_weights(mapping_id);

			for (int i = 0; i < node_ids.size(); ++i)
			{
				for (int k = 0; k < value_dim; ++k)
				{
					if (local_vec[k] == 0.0)
					{
						continue;
					}
					int offset = node_ids[i] * value_dim + k;
					assert(offset >= 0 && offset < global_vec.size());
					double val = node_weights[i] * local_vec[k];
					atomicAdd(global_vec.data() + offset, val);
				}
			}
		}

		template <typename ScalarKernel, int BLOCK_SIZE>
		__global__ void assemble_scalar_kernel(
			ScalarKernel kernel,
			AssemblyEssentialsView bases,
			AssemblyCacheView cache,
			int elem_num,
			Span<const typename ScalarKernel::Material> materials,
			Span<const double> unknown,
			double *scalar_out)
		{
			using Material = typename ScalarKernel::Material;

			assert(blockDim.x == BLOCK_SIZE);
			int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
			double scalar = 0.0; // local scalar value.

			if (elem_id < elem_num)
			{
				auto &cache_desc = cache.desc[elem_id];
				ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
				int quad_num = cache_desc.weighted_measure_range.num;

				for (int quad_id = 0; quad_id < quad_num; ++quad_id)
				{
					// Material is per cached quadrature point, matching weighted_measure.
					const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

					double local_scalar = kernel.eval_scalar(elem_id, quad_id, bases, elem_cache, material, unknown);
					scalar += local_scalar * elem_cache.get_weighted_measure(quad_id);
				}
			}

			using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
			__shared__ typename BlockReduce::TempStorage reduce_temp;
			double sum = BlockReduce(reduce_temp).Sum(scalar);
			if (threadIdx.x == 0)
			{
				atomicAdd(scalar_out, sum);
			}
		}

		template <typename ScalarKernel>
		__global__ void assemble_scalar_per_element_kernel(
			ScalarKernel kernel,
			AssemblyEssentialsView bases,
			AssemblyCacheView cache,
			Span<const typename ScalarKernel::Material> materials,
			Span<const double> unknown,
			Span<double> scalar_out)
		{
			using Material = typename ScalarKernel::Material;

			int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (elem_id >= scalar_out.size())
			{
				return;
			}

			auto &cache_desc = cache.desc[elem_id];
			ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
			int quad_num = cache_desc.weighted_measure_range.num;
			double scalar = 0.0; // local scalar value.

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				double local_scalar = kernel.eval_scalar(elem_id, quad_id, bases, elem_cache, material, unknown);
				scalar += local_scalar * elem_cache.get_weighted_measure(quad_id);
			}

			scalar_out[elem_id] += scalar;
		}

		template <typename VectorKernel>
		__global__ void assemble_vector_kernel(
			VectorKernel kernel,
			AssemblyEssentialsView bases,
			AssemblyCacheView cache,
			Span<const DeviceVectorAssemblyTask> tasks,
			Span<const typename VectorKernel::Material> materials,
			Span<const double> unknown,
			Span<double> vec_out,
			double extra_scaling)
		{
			constexpr int VALUE_DIM = VectorKernel::VALUE_DIM;

			using Material = typename VectorKernel::Material;
			using Vec = Eigen::Vector<double, VALUE_DIM>;

			int task_num = tasks.size();
			int task_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (task_id >= task_num)
			{
				return;
			}

			DeviceVectorAssemblyTask task = tasks[task_id];
			int elem_id = task.elem_id;
			int basis_id = task.basis_i;
			auto &cache_desc = cache.desc[elem_id];
			ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
			int quad_num = cache_desc.weighted_measure_range.num;
			Vec grad_i = Vec::Zero(); // local vector.

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				Vec kernel_out = kernel.eval_vector(elem_id, quad_id, basis_id, bases, elem_cache, material, unknown);
				grad_i += kernel_out * extra_scaling * elem_cache.get_weighted_measure(quad_id);
			}

			if (!grad_i.isZero())
			{
				Span<const double> grad_i_span(grad_i.data(), grad_i.size());
				scatter_vec_i<VALUE_DIM>(elem_id, basis_id, bases, grad_i_span, vec_out);
			}
		}

		template <typename MatrixKernel>
		__global__ void assemble_matrix_kernel(
			MatrixKernel kernel,
			AssemblyEssentialsView bases,
			AssemblyCacheView cache,
			Span<const DeviceMatrixAssemblyTask> tasks,
			Span<const typename MatrixKernel::Material> materials,
			Span<const double> unknown,
			BSRMatrixMutableView mat_out,
			double extra_scaling)
		{
			constexpr int VALUE_DIM = MatrixKernel::VALUE_DIM;

			using Material = typename MatrixKernel::Material;
			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;

			int task_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (task_id >= tasks.size())
			{
				return;
			}

			DeviceMatrixAssemblyTask task = tasks[task_id];
			int elem_id = task.elem_id;
			int bi = task.basis_i;
			int bj = task.basis_j;
			auto &cache_desc = cache.desc[elem_id];
			ElementAssemblyCacheView elem_cache = cache.slice(elem_id);
			int quad_num = cache_desc.weighted_measure_range.num;
			// ij component of element local matrix M.
			// Represents contribution from element local basis node i and j.
			Mat mat_ij = Mat::Zero();

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				Mat kernel_out = kernel.eval_matrix(elem_id, quad_id, bi, bj, bases, elem_cache, material, unknown);
				mat_ij += kernel_out * extra_scaling * elem_cache.get_weighted_measure(quad_id);
			}

			if (!mat_ij.isZero())
			{
				Span<const double> hess_ij_span(mat_ij.data(), mat_ij.size());
				scatter_mat_ij<VALUE_DIM>(elem_id, bi, bj, bases, hess_ij_span, mat_out);
				// We only eval upper half of M. Needs to scatters to lower part too.
				if (bj > bi)
				{
					Mat hess_ji = mat_ij.transpose();
					Span<const double> hess_ji_span(hess_ji.data(), hess_ji.size());
					scatter_mat_ij<VALUE_DIM>(elem_id, bj, bi, bases, hess_ji_span, mat_out);
				}
			}
		}

		template <typename Material, int dim>
		cuda::device_buffer<Material> prepare_materials(
			const AssemblyEssentials &bases,
			const AssemblyCache &cache,
			const material::MaterialExprRegistry &material_registry,
			double time,
			ExecutionPolicy policy)
		{
			auto cache_view = cache.view();

			int global_material_num = cache_view.weighted_measure.size();
			assert(global_material_num != 0);
			int elem_num = bases.element_desc.size();
			assert(elem_num != 0);

			if constexpr (std::is_same_v<Material, material::Dummy>)
			{
				auto d_materials =
					cuda::make_buffer<Material>(*policy.stream, *policy.mr, 0, cuda::no_init);
				policy.stream->sync();
				return d_materials;
			}
			else
			{
				std::vector<Material> materials(global_material_num);
				utils::maybe_parallel_for(elem_num, [&cache_view, &material_registry, &materials, time](int elem_id) {
					auto material_expr = material_registry.get<typename Material::ExprType>(elem_id);
					if (material_expr == nullptr)
					{
						throw std::runtime_error("Material missing!");
					}

					auto &cache_desc = cache_view.desc[elem_id];
					ElementAssemblyCacheView elem_cache = cache_view.slice(elem_id);
					for (int q = 0; q < cache_desc.weighted_measure_range.num; ++q)
					{
						double x = elem_cache.get_physical_x(q);
						double y = (dim >= 2) ? elem_cache.get_physical_y(q) : 0.0;
						double z = (dim >= 3) ? elem_cache.get_physical_z(q) : 0.0;
						Material m = material_expr->eval_expr(x, y, z, time, elem_id);

						materials[cache_desc.weighted_measure_range.offset + q] = std::move(m);
					}
				});

				auto d_materials =
					cuda::make_buffer<Material>(*policy.stream, *policy.mr, global_material_num, cuda::no_init);
				cuda::copy_bytes(*policy.stream, materials, d_materials);
				policy.stream->sync();
				return d_materials;
			}
		}
	} // namespace detail

	template <typename ScalarKernel>
	double assemble_scalar_on_device(
		ScalarKernel kernel,
		const AssemblyEssentials &bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		double time,
		ExecutionPolicy policy)
	{
		auto &p = policy;

		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			assert(!cache_desc.is_empty);
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, time, policy);
		auto d_unknown = cuda::make_buffer<double>(*p.stream, *p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(*p.stream, unknown, d_unknown);
		auto d_scalar_out = cuda::make_buffer<double>(*p.stream, *p.mr, 1, 0.0);

		int elem_num = bases.element_desc.size();
		int grid_num = div_round_up(elem_num, 128);
		detail::assemble_scalar_kernel<ScalarKernel, 128><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_bases,
			d_cache,
			elem_num,
			d_materials,
			d_unknown,
			d_scalar_out.data());
		double scalar_out = 0.0;
		cuda::copy_bytes(*p.stream, d_scalar_out, Span<double>(&scalar_out, 1));
		p.stream->sync();
		return scalar_out;
	}

	template <typename ScalarKernel>
	void assemble_scalar_per_element_on_device(
		ScalarKernel kernel,
		const AssemblyEssentials &bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vec_out,
		ExecutionPolicy policy,
		double time)
	{
		auto &p = policy;

		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			assert(!cache_desc.is_empty);
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, time, policy);

		int elem_num = bases.element_desc.size();
		assert(vec_out.size() == elem_num);

		auto d_unknown = cuda::make_buffer<double>(*p.stream, *p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(*p.stream, unknown, d_unknown);

		int grid_num = div_round_up(elem_num, 128);
		detail::assemble_scalar_per_element_kernel<ScalarKernel><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_bases,
			d_cache,
			d_materials,
			d_unknown,
			vec_out);
		p.stream->sync();
	}

	template <typename VectorKernel>
	void assemble_vector_on_device(
		VectorKernel kernel,
		const AssemblyEssentials &bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vec_out,
		ExecutionPolicy policy,
		double time = 0.0,
		double extra_scaling = 1.0)
	{
		auto &p = policy;

		auto d_vector_tasks = bases.device_vector_assembly_tasks(p);
		int task_num = d_vector_tasks.size();

		using Material = typename VectorKernel::Material;
		constexpr int DIM = VectorKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			assert(!cache_desc.is_empty);
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, time, policy);
		auto d_unknown = cuda::make_buffer<double>(*p.stream, *p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(*p.stream, unknown, d_unknown);

		int grid_num = div_round_up(task_num, 128);
		detail::assemble_vector_kernel<VectorKernel><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_bases,
			d_cache,
			d_vector_tasks,
			d_materials,
			d_unknown,
			vec_out,
			extra_scaling);
		p.stream->sync();
	}

	template <typename MatrixKernel>
	void assemble_matrix_on_device(
		MatrixKernel kernel,
		const AssemblyEssentials &bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		BSRMatrixMutableView mat_out,
		ExecutionPolicy policy,
		double time = 0.0,
		double extra_scaling = 1.0)
	{
		auto &p = policy;

		auto d_matrix_tasks = bases.device_matrix_assembly_tasks(p);
		int task_num = d_matrix_tasks.size();

		using Material = typename MatrixKernel::Material;
		constexpr int DIM = MatrixKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			assert(!cache_desc.is_empty);
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, time, policy);
		auto d_unknown = cuda::make_buffer<double>(*p.stream, *p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(*p.stream, unknown, d_unknown);

		int grid_num = div_round_up(task_num, 128);
		detail::assemble_matrix_kernel<MatrixKernel><<<grid_num, 128, 0, p.stream->get()>>>(
			kernel,
			d_bases,
			d_cache,
			d_matrix_tasks,
			d_materials,
			d_unknown,
			mat_out,
			extra_scaling);
		p.stream->sync();
	}

} // namespace polyfem::assembler
