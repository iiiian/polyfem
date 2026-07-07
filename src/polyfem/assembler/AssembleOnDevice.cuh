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

		POLYFEM_BOTH int index_upper_mat(int dim, int i, int j)
		{
			if (i > j)
			{
				cuda::std::swap(i, j);
			}
			return i * dim - (i * (i + 1) / 2) + j;
		}

		POLYFEM_BOTH cuda::std::pair<int, int> inverse_index_upper_mat(int dim, int offset)
		{
			for (int i = 0; i < dim; ++i)
			{
				for (int j = 0; j < dim; ++j)
				{
					if (index_upper_mat(dim, i, j) == offset)
						return {i, j};
				}
			}
			return {-1, -1};
		}

		// template <int VALUE_DIM>
		// __device__ void scatter_hess_psd_tmp(int local_i, int local_j, int basis_num, Span<const double> local_mat, Span<double> mat_out)
		// {
		// 	using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;
		// 	auto mat = Eigen::Map<const Mat>(local_mat.data());
		//
		// 	if (local_i != local_j)
		// 	{
		// 		for (int i = 0; i < VALUE_DIM; ++i)
		// 		{
		// 			for (int j = 0; j < VALUE_DIM; ++j)
		// 			{
		// 				int psd_i = VALUE_DIM * local_i + i;
		// 				int psd_j = VALUE_DIM * local_j + j;
		// 				mat_out[index_upper_mat(basis_num * VALUE_DIM, psd_i, psd_j)] = mat(i, j);
		// 			}
		// 		}
		// 	}
		// 	for (int i = 0; i < VALUE_DIM; ++i)
		// 	{
		// 		for (int j = i; j < VALUE_DIM; ++j)
		// 		{
		// 			int psd_i = VALUE_DIM * local_i + i;
		// 			int psd_j = VALUE_DIM * local_j + j;
		// 			mat_out[index_upper_mat(basis_num * VALUE_DIM, psd_i, psd_j)] = mat(i, j);
		// 		}
		// 	}
		// }

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

		// __device__ void scatter_mat_full(int elem_id, int basis_num, int value_dim, ElementBasesView bases, Span<const double> psd_mat, BSRMatrixMutableView mat_out)
		// {
		// 	auto &elem_desc = bases.element_desc[elem_id];
		// 	auto &mappings = bases.dof_mapping_store;
		//
		// 	for (int local_i = 0; local_i < basis_num; ++local_i)
		// 	{
		// 		for (int local_j = 0; local_j < basis_num; ++local_j)
		// 		{
		// 			int row_mapping_id = elem_desc.dof_mapping_range.offset + local_i;
		// 			int col_mapping_id = elem_desc.dof_mapping_range.offset + local_j;
		//
		// 			auto row_node_ids = mappings.get_node_ids(row_mapping_id);
		// 			auto row_node_weights = mappings.get_weights(row_mapping_id);
		// 			auto col_node_ids = mappings.get_node_ids(col_mapping_id);
		// 			auto col_node_weights = mappings.get_weights(col_mapping_id);
		//
		// 			for (int i = 0; i < row_node_ids.size(); ++i)
		// 			{
		// 				for (int j = 0; j < col_node_ids.size(); ++j)
		// 				{
		// 					double *block_ptr = mat_out.get_block(i, j);
		// 					assert(block_ptr);
		//
		// 					for (int vi = 0; vi < value_dim; ++vi)
		// 					{
		// 						for (int vj = 0; vj < value_dim; ++vj)
		// 						{
		// 							int psd_i = value_dim * local_i + vi;
		// 							int psd_j = value_dim * local_j + vj;
		// 							int psd_offset = index_upper_mat(basis_num * value_dim, psd_i, psd_j);
		// 							double val = row_node_weights[i] * col_node_weights[i] * psd_mat[psd_offset];
		// 							atomicAdd(block_ptr + index_upper_mat(value_dim, vi, vj), val);
		// 						}
		// 					}
		// 				}
		// 			}
		// 		}
		// 	}
		// }

		template <typename ScalarKernel>
		__global__ void assemble_scalar_kernel(
			ElementBasesView bases,
			AssemblyCacheView cache,
			Span<const typename ScalarKernel::Material> materials,
			Span<const double> unknown,
			double *scalar_out)
		{
			using Material = typename ScalarKernel::Material;

			assert(blockDim.x == 32);
			int elem_id = blockIdx.x;
			int warp_id = threadIdx.x;

			auto &cache_desc = cache.desc[elem_id];
			int quad_num = cache_desc.weighted_measure_range.num;
			double scalar = 0.0; // local scalar value.

			for (int quad_id = warp_id; quad_id < quad_num; quad_id += 32)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				double local_scalar = ScalarKernel::eval_scalar(elem_id, quad_id, bases, cache, material, unknown);
				scalar += local_scalar * cache.get_weighted_measure(elem_id, quad_id);
			}

			using BlockReduce = cub::BlockReduce<double, 32>;
			__shared__ typename BlockReduce::TempStorage reduce_temp;
			double sum = BlockReduce(reduce_temp).Sum(scalar);
			if (warp_id == 0)
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

			assert(blockDim.x == 32);
			int elem_id = blockIdx.x;
			int warp_id = threadIdx.x;

			auto &cache_desc = cache.desc[elem_id];
			int quad_num = cache_desc.weighted_measure_range.num;
			double scalar = 0.0; // local scalar value.

			for (int quad_id = warp_id; quad_id < quad_num; quad_id += 32)
			{
				// Material is per cached quadrature point, matching weighted_measure.
				const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

				double local_scalar = ScalarKernel::eval_scalar(elem_id, quad_id, bases, cache, material, unknown);
				scalar += local_scalar * cache.get_weighted_measure(elem_id, quad_id);
			}

			using BlockReduce = cub::BlockReduce<double, 32>;
			__shared__ typename BlockReduce::TempStorage reduce_temp;
			double sum = BlockReduce(reduce_temp).Sum(scalar);
			if (warp_id == 0)
			{
				scalar_out[elem_id] = sum;
			}
		}

		template <typename VectorKernel>
		__global__ void assemble_vector_kernel(
			ElementBasesView bases,
			AssemblyCacheView cache,
			Span<const typename VectorKernel::Material> materials,
			Span<const double> unknown,
			Span<double> vec_out)
		{
			constexpr int VALUE_DIM = VectorKernel::VALUE_DIM;

			using Material = typename VectorKernel::Material;
			using Vec = Eigen::Vector<double, VALUE_DIM>;

			assert(blockDim.x == 32);
			int elem_id = blockIdx.x;
			int warp_id = threadIdx.x;

			auto &elem_desc = bases.element_desc[elem_id];
			auto &basis_desc = elem_desc.basis_desc;
			int basis_num = basis_desc.basis_num;
			auto &cache_desc = cache.desc[elem_id];
			int quad_num = cache_desc.weighted_measure_range.num;

			using BlockReduce = cub::BlockReduce<Vec, 32>;
			__shared__ typename BlockReduce::TempStorage reduce_temp;

			for (int basis_id = 0; basis_id < basis_num; ++basis_id)
			{
				Vec grad_i = Vec::Zero(); // local vector.

				for (int quad_id = warp_id; quad_id < quad_num; quad_id += 32)
				{
					// Material is per cached quadrature point, matching weighted_measure.
					const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

					Vec local_grad_i = Vec::Zero();
					auto local_grad_i_span = Span<double>(local_grad_i.data(), local_grad_i.size());
					VectorKernel::eval_vector(elem_id, quad_id, basis_id, bases, cache, material, unknown, local_grad_i_span);
					grad_i += local_grad_i * cache.get_weighted_measure(elem_id, quad_id);
				}

				Vec sum = BlockReduce(reduce_temp).Sum(grad_i);

				if (warp_id == 0 && !sum.isZero())
				{
					Span<const double> sum_span(sum.data(), sum.size());
					scatter_vec_i<VALUE_DIM>(elem_id, basis_id, bases, sum_span, vec_out);
				}
			}
		}

		template <typename MatrixKernel>
		__global__ void assemble_matrix_kernel(
			ElementBasesView bases,
			AssemblyCacheView cache,
			Span<const typename MatrixKernel::Material> materials,
			Span<const double> unknown,
			BSRMatrixMutableView mat_out)
		{
			constexpr int VALUE_DIM = MatrixKernel::VALUE_DIM;

			using Material = typename MatrixKernel::Material;
			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;

			assert(blockDim.x == 32);
			int elem_id = blockIdx.x;
			int warp_id = threadIdx.x;

			auto &elem_desc = bases.element_desc[elem_id];
			auto &basis_desc = elem_desc.basis_desc;
			int basis_num = basis_desc.basis_num;
			auto &cache_desc = cache.desc[elem_id];
			int hessian_ij_num = upper_mat_size(basis_num);
			int quad_num = cache_desc.weighted_measure_range.num;

			// extern __shared__ double shared_psd_out[];
			// int psd_temp_size = upper_mat_size(basis_num);
			// auto psd_out_span = (psd_out) ? Span<double>(psd_out, psd_temp_size) : Span<double>(shared_psd_out, psd_temp_size);

			using BlockReduce = cub::BlockReduce<Mat, 32>;
			__shared__ typename BlockReduce::TempStorage reduce_temp;

			for (int ij = 0; ij < hessian_ij_num; ++ij)
			{
				auto [bi, bj] = inverse_index_upper_mat(basis_num, ij);
				Mat hess_ij = Mat::Zero(); // local Hij block.

				for (int quad_id = warp_id; quad_id < quad_num; quad_id += 32)
				{
					// Material is per cached quadrature point, matching weighted_measure.
					const Material &material = materials[cache_desc.weighted_measure_range.offset + quad_id];

					Mat local_hess_ij = Mat::Zero();
					auto hess_ij_span = Span<double>(local_hess_ij.data(), local_hess_ij.size());
					MatrixKernel::eval_matrix(elem_id, quad_id, bi, bj, bases, cache, material, unknown, hess_ij_span);
					hess_ij += local_hess_ij * cache.get_weighted_measure(elem_id, quad_id);
				}

				Mat sum = BlockReduce(reduce_temp).Sum(hess_ij);

				if (warp_id == 0 && !sum.isZero())
				{
					Span<const double> sum_span(sum.data(), sum.size());
					scatter_mat_ij<VALUE_DIM>(elem_id, bi, bj, bases, sum_span, mat_out);
					if (bj > bi)
					{
						Mat sum_t = sum.transpose();
						Span<const double> sum_t_span(sum_t.data(), sum_t.size());
						scatter_mat_ij<VALUE_DIM>(elem_id, bj, bi, bases, sum_t_span, mat_out);
					}
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

				const auto &cache_desc = cache_view.desc[elem_id];
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
		detail::assemble_scalar_kernel<ScalarKernel><<<elem_num, 32, 0, p.stream.get()>>>(
			d_bases,
			d_cache,
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

		detail::assemble_scalar_per_element_kernel<ScalarKernel><<<elem_num, 32, 0, p.stream.get()>>>(
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

		int elem_num = bases.element_desc.size();
		detail::assemble_vector_kernel<VectorKernel><<<elem_num, 32, 0, p.stream.get()>>>(
			d_bases,
			d_cache,
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

		int elem_num = bases.element_desc.size();
		detail::assemble_matrix_kernel<MatrixKernel><<<elem_num, 32, 0, p.stream.get()>>>(
			d_bases,
			d_cache,
			d_materials,
			d_unknown,
			mat_out);
		p.stream.sync();
	}

} // namespace polyfem::assembler
