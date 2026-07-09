#pragma once

#include <polyfem/assembler/AssembleOnHost.hpp>
#include <polyfem/utils/DualVector.hpp>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/assembler/AssembleOnDevice.cuh>
#endif

namespace polyfem::assembler
{
	namespace detail
	{
		inline bool cache_is_complete(
			const AssemblyEssentials &bases,
			const AssemblyCache &cache,
			bool is_mass)
		{
			AssemblyCacheView cache_view = cache.view();
			if (cache_view.desc.size() != bases.element_desc.size())
			{
				return false;
			}

			for (const AssemblyCacheDesc &desc : cache_view.desc)
			{
				if (desc.is_empty || desc.is_mass != is_mass)
					return false;
			}

			return true;
		}

		inline bool can_use_device(
			const AssemblyEssentials &bases,
			const AssemblyCache &cache,
			bool is_mass,
			ExecutionPolicy policy)
		{
#ifdef POLYFEM_WITH_CUDA
			return policy.mode == ExecutionMode::Hybrid
				   && policy.stream.has_value()
				   && policy.mr.has_value()
				   && cache_is_complete(bases, cache, is_mass);
#else
			(void)policy;
			(void)bases;
			(void)cache;
			(void)is_mass;
			return false;
#endif
		}
	} // namespace detail

	template <typename ScalarKernel>
	double assemble_scalar(
		const AssemblyEssentials &bases,
		const AssemblyEssentials &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		double time = 0.0,
		ExecutionPolicy policy = {})
	{
#ifdef POLYFEM_WITH_CUDA
		if (policy.mode == ExecutionMode::Hybrid && detail::cache_is_complete(bases, cache, false))
		{
			return assemble_scalar_on_device<ScalarKernel>(
				bases, cache, material_registry, unknown, time, policy);
		}
#endif

		return assemble_scalar_on_host<ScalarKernel>(
			bases, geom_bases, cache, material_registry, unknown, time);
	}

	template <typename ScalarKernel>
	void assemble_scalar_per_element(
		const AssemblyEssentials &bases,
		const AssemblyEssentials &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		DualVector &scalar_out,
		double time = 0.0,
		ExecutionPolicy policy = {})
	{
#ifdef POLYFEM_WITH_CUDA
		if (policy.mode == ExecutionMode::Hybrid && detail::cache_is_complete(bases, cache, false))
		{
			assemble_scalar_per_element_on_device<ScalarKernel>(
				bases,
				cache,
				material_registry,
				unknown,
				scalar_out.device_view(policy),
				policy,
				time);
			return;
		}
#endif

		assemble_scalar_per_element_on_host<ScalarKernel>(
			bases,
			geom_bases,
			cache,
			material_registry,
			unknown,
			scalar_out.host_view(),
			time);
	}

	template <typename VectorKernel>
	void assemble_vector(
		const AssemblyEssentials &bases,
		const AssemblyEssentials &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		DualVector &vector_out,
		double time = 0.0,
		double extra_scaling = 1.0,
		ExecutionPolicy policy = {})
	{
#ifdef POLYFEM_WITH_CUDA
		if (policy.mode == ExecutionMode::Hybrid && detail::cache_is_complete(bases, cache, false))
		{
			assemble_vector_on_device<VectorKernel>(
				bases,
				cache,
				material_registry,
				unknown,
				vector_out.device_view(policy),
				policy,
				time,
				extra_scaling);
			return;
		}
#endif

		assemble_vector_on_host<VectorKernel>(
			bases,
			geom_bases,
			cache,
			material_registry,
			unknown,
			vector_out.host_view(),
			time,
			extra_scaling);
	}

	template <typename MatrixKernel>
	void assemble_matrix(
		const AssemblyEssentials &bases,
		const AssemblyEssentials &geom_bases,
		const AssemblyCache &cache,
		const material::MaterialExprRegistry &material_registry,
		Span<const double> unknown,
		BSRMatrix &matrix_out,
		bool project_to_psd = false,
		double time = 0.0,
		double extra_scaling = 1.0,
		bool is_mass = false,
		ExecutionPolicy policy = {})
	{
#ifdef POLYFEM_WITH_CUDA
		if (policy.mode == ExecutionMode::Hybrid && !project_to_psd && detail::cache_is_complete(bases, cache, is_mass))
		{
			assemble_matrix_on_device<MatrixKernel>(
				bases,
				cache,
				material_registry,
				unknown,
				matrix_out.device_static_view(policy),
				policy,
				time,
				extra_scaling);
			return;
		}
#endif

		assemble_matrix_on_host<MatrixKernel>(
			bases,
			geom_bases,
			cache,
			material_registry,
			unknown,
			matrix_out.static_view(),
			project_to_psd,
			time,
			extra_scaling,
			is_mass);
	}
} // namespace polyfem::assembler
