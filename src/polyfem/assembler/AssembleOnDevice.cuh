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
					double *block_ptr = global_mat.get_block(i, j);
					assert(block_ptr);

					for (int k = 0; k < mat.size(); ++k)
					{
						double val = row_node_weights[i] * col_node_weights[i] * mat.data()[k];
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
				for (int k = 0; k < global_vec.size(); ++k)
				{
					int offset = node_ids[i] * value_dim + k;
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

			auto &elem_desc = bases.element_desc[warp_id];
			auto &quad_desc = elem_desc.quadrature_desc;
			int quad_num = quad_desc.w_range.num;
			int max_loop_num = quad_num;
			int wave_num = div_round_up(max_loop_num, 32);

			for (int w = 0; w < wave_num; ++w)
			{
				int loop_id = w * 32 + warp_id;
				int quad_id = loop_id;
				double scalar = 0.0; // local scalar value.

				if (loop_id < max_loop_num)
				{
					// Material is per quadrature, thus the indexing rule is the same as quadrature weight.
					const Material &material = materials[quad_desc.w_range.offset + quad_id];

					scalar = ScalarKernel::eval_scalar(elem_id, quad_id, bases, cache, material, unknown);
					scalar *= cache.get_weighted_measure(elem_id, quad_id);
				}

				using BlockReduce = cub::BlockReduce<double, 32>;
				__shared__ typename BlockReduce::TempStorage reduce_temp;
				double sum = BlockReduce(reduce_temp).Sum(scalar);
				if (warp_id == 0)
				{
					atomicAdd(scalar_out, sum);
				}
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

			auto &elem_desc = bases.element_desc[warp_id];
			auto &quad_desc = elem_desc.quadrature_desc;
			int quad_num = quad_desc.w_range.num;
			int max_loop_num = quad_num;
			int wave_num = div_round_up(max_loop_num, 32);

			for (int w = 0; w < wave_num; ++w)
			{
				int loop_id = w * 32 + warp_id;
				int quad_id = loop_id;
				double scalar = 0.0; // local scalar value.

				if (loop_id < max_loop_num)
				{
					// Material is per quadrature, thus the indexing rule is the same as quadrature weight.
					const Material &material = materials[quad_desc.w_range.offset + quad_id];

					scalar = ScalarKernel::eval_scalar(elem_id, quad_id, bases, cache, material, unknown);
					scalar *= cache.get_weighted_measure(elem_id, quad_id);
				}

				using BlockReduce = cub::BlockReduce<double, 32>;
				__shared__ typename BlockReduce::TempStorage reduce_temp;
				double sum = BlockReduce(reduce_temp).Sum(scalar);
				if (warp_id == 0)
				{
					scalar_out[elem_id] = sum;
				}
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

			auto &elem_desc = bases.element_desc[warp_id];
			auto &basis_desc = elem_desc.basis_desc;
			int basis_num = basis_desc.basis_num;
			auto &quad_desc = elem_desc.quadrature_desc;
			int quad_num = quad_desc.w_range.num;
			int max_loop_num = basis_num * quad_num;
			int wave_num = div_round_up(max_loop_num, 32);

			for (int w = 0; w < wave_num; ++w)
			{
				int loop_id = w * 32 + warp_id;
				int quad_id = loop_id % basis_num;
				int basis_id = loop_id / basis_num;
				Vec grad_i = Vec::Zero(); // local vector.

				if (loop_id < max_loop_num)
				{
					// Material is per quadrature, thus the indexing rule is the same as quadrature weight.
					const Material &material = materials[quad_desc.w_range.offset + quad_id];

					VectorKernel::eval_vector(elem_id, quad_id, basis_id, bases, cache, material, unknown, grad_i);
					grad_i *= cache.get_weighted_measure(elem_id, quad_id);
				}

				// Wrap reduction, sum local gradient Gi from all quadrature points before scattering.
				bool is_head = (loop_id < max_loop_num) && ((warp_id == 0) || (quad_id == 0));
				using WarpReduce = cub::WarpReduce<Vec>;
				__shared__ typename WarpReduce::TempStorage reduce_temp;
				Vec sum = WarpReduce(reduce_temp).HeadSegmentedSum(grad_i, is_head);

				if (is_head && !grad_i.isZero())
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
			using Vec = Eigen::Vector<double, VALUE_DIM>;
			using Mat = Eigen::Matrix<double, VALUE_DIM, VALUE_DIM, Eigen::RowMajor>;

			assert(blockDim.x == 32);
			int elem_id = blockIdx.x;
			int warp_id = threadIdx.x;

			auto &elem_desc = bases.element_desc[warp_id];
			auto &basis_desc = elem_desc.basis_desc;
			int basis_num = basis_desc.basis_num;
			auto &quad_desc = elem_desc.quadrature_desc;
			int hessian_ij_num = upper_mat_size(basis_num);
			int quad_num = quad_desc.w_range.num;
			int max_loop_num = hessian_ij_num * quad_num;
			int wave_num = div_round_up(max_loop_num, 32);

			// extern __shared__ double shared_psd_out[];
			// int psd_temp_size = upper_mat_size(basis_num);
			// auto psd_out_span = (psd_out) ? Span<double>(psd_out, psd_temp_size) : Span<double>(shared_psd_out, psd_temp_size);

			for (int w = 0; w < wave_num; ++w)
			{
				int loop_id = w * 32 + warp_id;
				int quad_id = loop_id % hessian_ij_num;
				int ij = loop_id / hessian_ij_num;
				auto [bi, bj] = inverse_index_upper_mat(basis_num, ij);
				Mat hess_ij = Mat::Zero(); // local Hij block.

				if (loop_id < max_loop_num)
				{
					// Material is per quadrature, thus the indexing rule is the same as quadrature weight.
					const Material &material = materials[quad_desc.w_range.offset + quad_id];

					auto hess_ij_span = Span<double>(hess_ij.data(), hess_ij.size());
					MatrixKernel::eval_matrix(elem_id, quad_id, bi, bj, bases, cache, material, unknown, hess_ij_span);
					hess_ij *= cache.get_weighted_measure(elem_id, quad_id);
				}

				// Wrap reduction, sum local hessian Hij from all quadrature points before scattering.
				bool is_head = (loop_id < max_loop_num) && ((warp_id == 0) || (quad_id == 0));
				using WarpReduce = cub::WarpReduce<Mat>;
				__shared__ typename WarpReduce::TempStorage reduce_temp;
				Mat sum = WarpReduce(reduce_temp).HeadSegmentedSum(hess_ij, is_head);

				if (is_head && !hess_ij.isZero())
				{
					Span<const double> sum_span(sum.data(), sum.size());
					scatter_mat_ij<VALUE_DIM>(elem_id, bi, bj, bases, sum_span, mat_out);
					if (bj > bi)
					{
						scatter_mat_ij<VALUE_DIM>(elem_id, bj, bi, bases, sum_span, mat_out);
					}
				}
			}
		}

		template <typename Material, int dim>
		cuda::device_buffer<Material> prepare_meterials(
			const ElementBases &bases,
			const AssemblyCache &cache,
			const material::MaterialExprRegistry &material_registry,
			CudaExecutionPolicy policy)
		{
			int quad_num = bases.quadrature_store.view().w.size();
			assert(quad_num != 0);
			int elem_num = bases.element_desc.size();
			assert(elem_num != 0);

			std::vector<Material> materials(quad_num);
			utils::maybe_parallel_for(elem_num, [quad_num, &material_registry, &cache, &materials](int elem_id) {
				for (int q = 0; q < quad_num; ++q)
				{
					auto material_expr = material_registry.get<typename Material::ExprType>(elem_id);
					if (material_expr == nullptr)
					{
						throw std::runtime_error("Material missing!");
					}

					auto cache_view = cache.view();
					double x = cache_view.get_physical_x(elem_id, q);
					double y = (dim >= 2) ? cache_view.get_physical_y(elem_id, q) : 0.0;
					double z = (dim >= 3) ? cache_view.get_physical_z(elem_id, q) : 0.0;
					Material m = material_expr->eval_expr(x, y, z, 0, elem_id);

					materials[q] = std::move(m);
				}
			});
			auto d_materials =
				cuda::make_buffer<Material>(policy.stream, policy.mr, quad_num, cuda::no_init);
			cuda::copy_bytes(policy.stream, materials, d_materials);
			policy.stream.sync();
			return d_materials;
		}
	} // namespace detail

	template <typename ScalarKernel>
	double assemble_scalar(
		const ElementBases &bases,
		const AssemblyCache &cache,
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

		auto d_materials = detail::prepare_meterials<Material, DIM>(bases, cache, material_registry, policy);
		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);
		auto d_scalar_out = cuda::make_buffer<double>(p.stream, p.mr, 1, 0.0);

		int elem_num = bases.element_desc.size();
		detail::assemble_scalar_kernel<ScalarKernel><<<elem_num, 32, 0, p.stream.get()>>>(
			bases.view(),
			cache.view(),
			d_materials,
			d_unknown,
			d_scalar_out.data());
		p.stream.sync();
	}

	template <typename ScalarKernel>
	void assemble_scalar_per_element(
		const ElementBases &bases,
		const AssemblyCache &cache,
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

		auto d_materials = detail::prepare_meterials<Material, DIM>(bases, cache, material_registry, policy);
		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);

		int elem_num = bases.element_desc.size();
		detail::assemble_scalar_per_element_kernel<ScalarKernel><<<elem_num, 32, 0, p.stream.get()>>>(
			bases.view(),
			cache.view(),
			d_materials,
			d_unknown,
			vec_out.data());
		p.stream.sync();
	}

	template <typename VectorKernel>
	void assemble_vector(
		const ElementBases &bases,
		const AssemblyCache &cache,
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

		auto d_materials = detail::prepare_meterials<Material, DIM>(bases, cache, material_registry, policy);
		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);

		int elem_num = bases.element_desc.size();
		detail::assemble_vector_kernel<VectorKernel><<<elem_num, 32, 0, p.stream.get()>>>(
			bases.view(),
			cache.view(),
			d_materials,
			d_unknown,
			vec_out);
		p.stream.sync();
	}

	template <typename MatrixKernel>
	void assemble_matrix(
		const ElementBases &bases,
		const AssemblyCache &cache,
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

		auto d_materials = detail::prepare_meterials<Material, DIM>(bases, cache, material_registry, policy);
		auto d_unknown = cuda::make_buffer<double>(p.stream, p.mr, unknown.size(), cuda::no_init);
		cuda::copy_bytes(p.stream, unknown, d_unknown);

		int elem_num = bases.element_desc.size();
		detail::assemble_matrix_kernel<MatrixKernel><<<elem_num, 32, 0, p.stream.get()>>>(
			bases.view(),
			cache.view(),
			d_materials,
			d_unknown,
			mat_out);
		p.stream.sync();
	}

} // namespace polyfem::assembler
