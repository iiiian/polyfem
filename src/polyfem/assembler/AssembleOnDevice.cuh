#pragma once

#include <polyfem/materials/MaterialExprRegistry.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/assembler/ElementBases.hpp>
#include <polyfem/utils/CUDAExecutionPolicy.hpp>
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
#include <stdexcept>
#include <vector>
#include <exception>

namespace polyfem::assembler
{
	namespace detail
	{

		POLYFEM_BOTH int upper_mat_size(int dim)
		{
			return dim * (dim + 1) / 2;
		}

		struct VectorAssemblyTask
		{
			int elem_id;
			int local_i;
		};

		struct MatrixAssemblyTask
		{
			int elem_id;
			int local_i;
			int local_j;
		};

		inline std::vector<VectorAssemblyTask> build_vector_tasks(const ElementBases &bases)
		{
			int task_num = 0;
			for (const ElementDesc &elem_desc : bases.element_desc)
			{
				task_num += elem_desc.basis_desc.basis_num;
			}

			std::vector<VectorAssemblyTask> tasks;
			tasks.reserve(task_num);
			for (int elem_id = 0; elem_id < bases.element_desc.size(); ++elem_id)
			{
				int basis_num = bases.element_desc[elem_id].basis_desc.basis_num;
				for (int local_i = 0; local_i < basis_num; ++local_i)
				{
					tasks.push_back({elem_id, local_i});
				}
			}

			return tasks;
		}

		inline std::vector<MatrixAssemblyTask> build_matrix_tasks(const ElementBases &bases)
		{
			int task_num = 0;
			for (const ElementDesc &elem_desc : bases.element_desc)
			{
				task_num += upper_mat_size(elem_desc.basis_desc.basis_num);
			}

			std::vector<MatrixAssemblyTask> tasks;
			tasks.reserve(task_num);
			for (int elem_id = 0; elem_id < bases.element_desc.size(); ++elem_id)
			{
				int basis_num = bases.element_desc[elem_id].basis_desc.basis_num;
				for (int local_i = 0; local_i < basis_num; ++local_i)
				{
					for (int local_j = local_i; local_j < basis_num; ++local_j)
					{
						tasks.push_back({elem_id, local_i, local_j});
					}
				}
			}

			return tasks;
		}

		template <int VALUE_DIM>
		__device__ void scatter_mat_ij(
			int elem_id,
			int local_i,
			int local_j,
			ElementBasesView bases,
			Span<const double> local_mat,
			BSRMatrixMutableView global_mat)
		{
			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;
			auto mat = Eigen::Map<const Mat>(local_mat.data());
			assert(global_mat.block_dim == VALUE_DIM);

			auto &elem_desc = bases.element_desc[elem_id];
			auto &mappings = bases.dof_mapping_store;
			int row_mapping_id = elem_desc.dof_mapping_range.offset + local_i;
			int col_mapping_id = elem_desc.dof_mapping_range.offset + local_j;

			auto row_node_ids = mappings.get_node_ids(row_mapping_id);
			auto row_node_weights = mappings.get_weights(row_mapping_id);
			auto col_node_ids = mappings.get_node_ids(col_mapping_id);
			auto col_node_weights = mappings.get_weights(col_mapping_id);

			for (int i = 0; i < row_node_ids.size(); ++i)
			{
				for (int j = 0; j < col_node_ids.size(); ++j)
				{
					double *block_ptr = global_mat.get_block(row_node_ids[i], col_node_ids[j]);
					assert(block_ptr);

					for (int k = 0; k < mat.size(); ++k)
					{
						double val = row_node_weights[i] * col_node_weights[j] * mat.data()[k];
						atomicAdd(block_ptr + k, val);
					}
				}
			}
		}

		template <int value_dim>
		__device__ void scatter_vec_i(
			int elem_id,
			int local_i,
			ElementBasesView bases,
			Span<const double> local_vec,
			Span<double> global_vec)
		{
			assert(local_vec.size() == value_dim);

			auto &elem_desc = bases.element_desc[elem_id];
			auto &mappings = bases.dof_mapping_store;
			int mapping_id = elem_desc.dof_mapping_range.offset + local_i;

			auto node_ids = mappings.get_node_ids(mapping_id);
			auto node_weights = mappings.get_weights(mapping_id);

			for (int i = 0; i < node_ids.size(); ++i)
			{
				for (int k = 0; k < value_dim; ++k)
				{
					int offset = node_ids[i] * value_dim + k;
					assert(offset >= 0 && offset < global_vec.size());
					double val = node_weights[i] * local_vec[k];
					atomicAdd(global_vec.data() + offset, val);
				}
			}
		}

		template <typename ScalarKernel, int BLOCK_SIZE>
		__global__ void assemble_scalar_kernel(
			ElementBasesView bases,
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
				int quad_num = cache_desc.weighted_measure_range.num;

				for (int quad_id = 0; quad_id < quad_num; ++quad_id)
				{
					// Material is per cached quadrature point, matching weighted_measure.
					const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

					double local_scalar = ScalarKernel::eval_scalar(elem_id, quad_id, bases, cache, material, unknown);
					scalar += local_scalar * cache.get_weighted_measure(elem_id, quad_id);
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
			ElementBasesView bases,
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
			int quad_num = cache_desc.weighted_measure_range.num;
			double scalar = 0.0; // local scalar value.

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				double local_scalar = ScalarKernel::eval_scalar(elem_id, quad_id, bases, cache, material, unknown);
				scalar += local_scalar * cache.get_weighted_measure(elem_id, quad_id);
			}

			scalar_out[elem_id] += scalar;
		}

		template <typename VectorKernel>
		__global__ void assemble_vector_kernel(
			ElementBasesView bases,
			AssemblyCacheView cache,
			Span<const VectorAssemblyTask> tasks,
			Span<const typename VectorKernel::Material> materials,
			Span<const double> unknown,
			Span<double> vec_out)
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

			VectorAssemblyTask task = tasks[task_id];
			int elem_id = task.elem_id;
			int basis_id = task.local_i;
			auto &cache_desc = cache.desc[elem_id];
			int quad_num = cache_desc.weighted_measure_range.num;
			Vec grad_i = Vec::Zero(); // local vector.

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				Vec local_grad_i = Vec::Zero();
				auto local_grad_i_span = Span<double>(local_grad_i.data(), local_grad_i.size());
				VectorKernel::eval_vector(elem_id, quad_id, basis_id, bases, cache, material, unknown, local_grad_i_span);
				grad_i += local_grad_i * cache.get_weighted_measure(elem_id, quad_id);
			}

			if (!grad_i.isZero())
			{
				Span<const double> grad_i_span(grad_i.data(), grad_i.size());
				scatter_vec_i<VALUE_DIM>(elem_id, basis_id, bases, grad_i_span, vec_out);
			}
		}

		template <typename MatrixKernel>
		__global__ void assemble_matrix_kernel(
			ElementBasesView bases,
			AssemblyCacheView cache,
			Span<const MatrixAssemblyTask> tasks,
			Span<const typename MatrixKernel::Material> materials,
			Span<const double> unknown,
			BSRMatrixMutableView mat_out)
		{
			constexpr int VALUE_DIM = MatrixKernel::VALUE_DIM;

			using Material = typename MatrixKernel::Material;
			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;

			const int task_num = static_cast<int>(tasks.size());
			const int task_id = blockIdx.x * blockDim.x + threadIdx.x;
			if (task_id >= task_num)
			{
				return;
			}

			MatrixAssemblyTask task = tasks[task_id];
			int elem_id = task.elem_id;
			int bi = task.local_i;
			int bj = task.local_j;
			auto &cache_desc = cache.desc[elem_id];
			int quad_num = cache_desc.weighted_measure_range.num;
			Mat hess_ij = Mat::Zero(); // local Hij block.

			for (int quad_id = 0; quad_id < quad_num; ++quad_id)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				Mat local_hess_ij = Mat::Zero();
				auto hess_ij_span = Span<double>(local_hess_ij.data(), local_hess_ij.size());
				MatrixKernel::eval_matrix(elem_id, quad_id, bi, bj, bases, cache, material, unknown, hess_ij_span);
				hess_ij += local_hess_ij * cache.get_weighted_measure(elem_id, quad_id);
			}

			if (!hess_ij.isZero())
			{
				Span<const double> hess_ij_span(hess_ij.data(), hess_ij.size());
				scatter_mat_ij<VALUE_DIM>(elem_id, bi, bj, bases, hess_ij_span, mat_out);
				if (bj > bi)
				{
					Mat hess_ji = hess_ij.transpose();
					Span<const double> hess_ji_span(hess_ji.data(), hess_ji.size());
					scatter_mat_ij<VALUE_DIM>(elem_id, bj, bi, bases, hess_ji_span, mat_out);
				}
			}
		}

		template <typename Material, int dim>
		cuda::device_buffer<Material> prepare_materials(
			const ElementBases &bases,
			const AssemblyCache &cache,
			const material::MaterialExprRegistry &material_registry,
			CudaExecutionPolicy policy)
		{
			auto cache_view = cache.view();

			int global_material_num = cache_view.weighted_measure.size();
			assert(global_material_num != 0);
			int elem_num = bases.element_desc.size();
			assert(elem_num != 0);

			std::vector<Material> materials(global_material_num);
			utils::maybe_parallel_for(elem_num, [&cache_view, &material_registry, &materials](int elem_id) {
				auto material_expr = material_registry.get<typename Material::ExprType>(elem_id);
				if (material_expr == nullptr)
				{
					throw std::runtime_error("Material missing!");
				}

				auto &cache_desc = cache_view.desc[elem_id];
				for (int q = 0; q < cache_desc.weighted_measure_range.num; ++q)
				{
					double x = cache_view.get_physical_x(elem_id, q);
					double y = (dim >= 2) ? cache_view.get_physical_y(elem_id, q) : 0.0;
					double z = (dim >= 3) ? cache_view.get_physical_z(elem_id, q) : 0.0;
					Material m = material_expr->eval_expr(x, y, z, 0, elem_id);

					materials[cache_desc.weighted_measure_range.offset + q] = std::move(m);
				}
			});
			auto d_materials =
				cuda::make_buffer<Material>(policy.stream, policy.mr, global_material_num, cuda::no_init);
			cuda::copy_bytes(policy.stream, materials, d_materials);
			policy.stream.sync();
			return d_materials;
		}
	} // namespace detail

	template <typename ScalarKernel>
	double assemble_scalar(
		ElementBases &bases,
		AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		CudaExecutionPolicy policy = {})
	{
		auto &p = policy;

		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			if (cache_desc.is_empty)
			{
				throw std::runtime_error("Cuda assembler requires pre-computed assembly cache.");
			}
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, policy);
		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);
		auto d_scalar_out = cuda::make_buffer<double>(p.stream, p.mr, 1, 0.0);

		int elem_num = bases.element_desc.size();
		int grid_num = div_round_up(elem_num, 128);
		detail::assemble_scalar_kernel<ScalarKernel, 128><<<grid_num, 128, 0, p.stream.get()>>>(
			d_bases,
			d_cache,
			elem_num,
			d_materials,
			d_unknown,
			d_scalar_out.data());
		double scalar_out = 0.0;
		cuda::copy_bytes(p.stream, d_scalar_out, Span<double>(&scalar_out, 1));
		p.stream.sync();
		return scalar_out;
	}

	template <typename ScalarKernel>
	void assemble_scalar_per_element(
		ElementBases &bases,
		AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vec_out,
		CudaExecutionPolicy policy = {})
	{
		auto &p = policy;

		using Material = typename ScalarKernel::Material;
		constexpr int DIM = ScalarKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			if (cache_desc.is_empty)
			{
				throw std::runtime_error("Cuda assembler requires pre-computed assembly cache.");
			}
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, policy);

		int elem_num = bases.element_desc.size();
		assert(vec_out.size() == elem_num);

		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);

		int grid_num = div_round_up(elem_num, 128);
		detail::assemble_scalar_per_element_kernel<ScalarKernel><<<grid_num, 128, 0, p.stream.get()>>>(
			d_bases,
			d_cache,
			d_materials,
			d_unknown,
			vec_out);
		p.stream.sync();
	}

	template <typename VectorKernel>
	void assemble_vector(
		ElementBases &bases,
		AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		Span<double> vec_out,
		CudaExecutionPolicy policy = {})
	{
		auto &p = policy;

		using Material = typename VectorKernel::Material;
		constexpr int DIM = VectorKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			if (cache_desc.is_empty)
			{
				throw std::runtime_error("Cuda assembler requires pre-computed assembly cache.");
			}
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, policy);
		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);
		auto vector_tasks = detail::build_vector_tasks(bases);
		const int task_num = static_cast<int>(vector_tasks.size());
		assert(static_cast<std::size_t>(task_num) == vector_tasks.size());
		auto d_vector_tasks = cuda::make_buffer<detail::VectorAssemblyTask>(
			p.stream,
			p.mr,
			vector_tasks.size(),
			cuda::no_init);
		cuda::copy_bytes(p.stream, vector_tasks, d_vector_tasks);

		int grad_num = div_round_up(task_num, 128);
		detail::assemble_vector_kernel<VectorKernel><<<grid_num, 128, 0, p.stream.get()>>>(
			d_bases,
			d_cache,
			d_vector_tasks,
			d_materials,
			d_unknown,
			vec_out);
		p.stream.sync();
	}

	template <typename MatrixKernel>
	void assemble_matrix(
		ElementBases &bases,
		AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		BSRMatrixMutableView mat_out,
		CudaExecutionPolicy policy = {})
	{
		auto &p = policy;

		using Material = typename MatrixKernel::Material;
		constexpr int DIM = MatrixKernel::DIM;

		for (auto &cache_desc : cache.view().desc)
		{
			if (cache_desc.is_empty)
			{
				throw std::runtime_error("Cuda assembler requires pre-computed assembly cache.");
			}
		}

		auto d_bases = bases.device_view(p);
		auto d_cache = cache.device_view(p);
		auto d_materials = detail::prepare_materials<Material, DIM>(bases, cache, material_registry, policy);
		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);
		auto matrix_tasks = detail::build_matrix_tasks(bases);
		const int task_num = static_cast<int>(matrix_tasks.size());
		assert(static_cast<std::size_t>(task_num) == matrix_tasks.size());
		auto d_matrix_tasks = cuda::make_buffer<detail::MatrixAssemblyTask>(
			p.stream,
			p.mr,
			matrix_tasks.size(),
			cuda::no_init);
		cuda::copy_bytes(p.stream, matrix_tasks, d_matrix_tasks);

		if (task_num > 0)
		{
			int grad_num = div_round_up(task_num, 128);
			detail::assemble_matrix_kernel<MatrixKernel><<<grid_num, 128, 0, p.stream.get()>>>(
				d_bases,
				d_cache,
				d_matrix_tasks,
				d_materials,
				d_unknown,
				mat_out);
		}
		p.stream.sync();
	}

} // namespace polyfem::assembler
