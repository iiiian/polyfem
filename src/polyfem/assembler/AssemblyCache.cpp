#include <polyfem/assembler/AssemblyCache.hpp>

#include <algorithm>
#include <cassert>
#include <Eigen/Core>

namespace polyfem::assembler
{
	namespace
	{

		template <typename T>
		Range append(int num, std::vector<T> &vec)
		{
			vec.resize(vec.size() + num);
			return Range{static_cast<int>(vec.size() - num), num};
		}

		AssemblyCacheDesc resize_cache_storage(
			int dim,
			int quad_num,
			int basis_num,
			int geom_basis_num,
			bool need_basis_values,
			bool needs_basis_gradients,
			bool needs_geom_mapping_scratch,
			GeomMappingScratch &geom_scratch,
			AssemblyCache &cache_store)
		{
			auto &s = cache_store;

			AssemblyCacheDesc desc;

			int basis_storage_size = basis_num * quad_num;
			if (need_basis_values)
			{
				desc.basis_val_range = append(basis_storage_size, s.basis_values);
			}
			if (needs_basis_gradients)
			{
				// clang-format off
        if (dim >= 1) desc.basis_grad_x_range = append(basis_storage_size, s.basis_grad_x);
        if (dim >= 2) desc.basis_grad_y_range = append(basis_storage_size, s.basis_grad_y);
        if (dim >= 3) desc.basis_grad_z_range = append(basis_storage_size, s.basis_grad_z);
        if (dim >= 1) desc.basis_grad_phy_x_range = append(basis_storage_size, s.basis_grad_phy_x);
        if (dim >= 2) desc.basis_grad_phy_y_range = append(basis_storage_size, s.basis_grad_phy_y);
        if (dim >= 3) desc.basis_grad_phy_z_range = append(basis_storage_size, s.basis_grad_phy_z);
				// clang-format on
			}

			int physical_storage_size = quad_num;
			// clang-format off
			if (dim >= 1) desc.physical_x_range = append(physical_storage_size, s.physical_x);
			if (dim >= 2) desc.physical_y_range = append(physical_storage_size, s.physical_y);
			if (dim >= 3) desc.physical_z_range = append(physical_storage_size, s.physical_z);
			// clang-format on

			desc.det_J_range = append(quad_num, s.det_J);
			desc.J_inverse_transpose_range = append(quad_num * dim * dim, s.J_inverse_transpose);
			desc.weighted_measure_range = append(quad_num, s.weighted_measure);

			if (needs_geom_mapping_scratch)
			{
				int geom_basis_storage_size = geom_basis_num * quad_num;
				// clang-format off
        geom_scratch.basis_values.resize(geom_basis_storage_size);
        if (dim >= 1) geom_scratch.basis_grad_x.resize(geom_basis_storage_size);
        if (dim >= 2) geom_scratch.basis_grad_y.resize(geom_basis_storage_size);
        if (dim >= 3) geom_scratch.basis_grad_z.resize(geom_basis_storage_size);
				// clang-format on
			}

			return desc;
		}

		void set_identity_J_inverse_transpose(int dim, int quad_num, span<double> out)
		{
			std::fill(out.begin(), out.end(), 0);
			for (int q = 0; q < quad_num; ++q)
			{
				for (int d = 0; d < dim; ++d)
					out[q * dim * dim + d * dim + d] = 1;
			}
		}

		void compute_weighted_measure(span<const double> quad_w, span<const double> det_J, span<double> out)
		{
			for (int q = 0; q < quad_w.size(); ++q)
			{
				out[q] = quad_w[q] * det_J[q];
			}
		}

		void compute_basis_grad_phy(
			int dim,
			int basis_num,
			int quad_num,
			span<const double> grad_x,
			span<const double> grad_y,
			span<const double> grad_z,
			span<const double> J_it,
			span<double> grad_phy_x,
			span<double> grad_phy_y,
			span<double> grad_phy_z)
		{
			// int n_entries = n_bases * n_points;

			std::array<span<const double>, 3> grad = {{grad_x, grad_y, grad_z}};
			std::array<span<double>, 3> grad_phy = {{grad_phy_x, grad_phy_y, grad_phy_z}};

			for (int bi = 0; bi < basis_num; ++bi)
			{
				for (int qi = 0; qi < quad_num; ++qi)
				{
					int entry = bi * quad_num + qi;
					auto *jac_it = J_it.data() + qi * dim * dim;

					for (int d = 0; d < dim; ++d)
					{
						double value = 0;
						for (int r = 0; r < dim; ++r)
							value += grad[r][entry] * jac_it[r * dim + d];
						grad_phy[d][entry] = value;
					}
				}
			}
		}

	} // namespace

	AssemblyCacheView AssemblyCache::view() const
	{
		return AssemblyCacheView{
			desc,
			basis_values,
			basis_grad_x,
			basis_grad_y,
			basis_grad_z,
			basis_grad_phy_x,
			basis_grad_phy_y,
			basis_grad_phy_z,
			physical_x,
			physical_y,
			physical_z,
			det_J,
			J_inverse_transpose,
			weighted_measure};
	}

	void AssemblyCache::reset()
	{
		desc.resize(0);
		basis_values.resize(0);
		basis_grad_x.resize(0);
		basis_grad_y.resize(0);
		basis_grad_z.resize(0);
		basis_grad_phy_x.resize(0);
		basis_grad_phy_y.resize(0);
		basis_grad_phy_z.resize(0);
		physical_x.resize(0);
		physical_y.resize(0);
		physical_z.resize(0);
		det_J.resize(0);
		J_inverse_transpose.resize(0);
		weighted_measure.resize(0);
	};

	void compute_assembly_cache_single(
		const basis::ng::ElementBasesView &bases,
		const basis::ng::ElementBasesView &geom_bases,
		int element_id,
		bool is_mass,
		bool is_isoparametric,
		bool need_basis_values,
		bool need_basis_gradients,
		GeomMappingScratch &geom_mapping_scratch,
		AssemblyCache &cache)
	{
		assert(0 <= element_id && element_id < bases.element_desc.size());
		assert(0 <= element_id && element_id < geom_bases.element_desc.size());

		auto &element_desc = bases.element_desc[element_id];
		auto &geom_element_desc = geom_bases.element_desc[element_id];
		auto &quad_desc = is_mass ? element_desc.mass_quadrature_desc : element_desc.quadrature_desc;
		auto &quad_store = is_mass ? bases.mass_quadrature : bases.quadrature;

		int dim = element_desc.basis_desc.dim;
		int quad_num = quad_desc.w_range.num;
		int basis_num = basis::ng::local_basis_num(element_desc.basis_desc);
		int geom_basis_num = basis::ng::local_basis_num(geom_element_desc.basis_desc);

		// Physical xyz and geom mapping fields are required and depends on geom basis value and grad.
		// So in isoparametric case (geom basis == basis) we need to eval basis and basis gradient
		// not matter what.
		need_basis_values = is_isoparametric ? true : need_basis_values;
		need_basis_gradients = is_isoparametric ? true : need_basis_gradients;
		auto desc = resize_cache_storage(
			dim,
			quad_num,
			basis_num,
			geom_basis_num,
			need_basis_values,
			need_basis_gradients,
			!is_isoparametric,
			geom_mapping_scratch,
			cache);
		desc.is_empty = false;
		desc.is_mass = is_mass;
		cache.desc.push_back(desc);

		// Quadrature point and weight.
		auto quad_x = slice_by_range(quad_store.x, quad_desc.x_range);
		auto quad_y = slice_by_range(quad_store.y, quad_desc.y_range);
		auto quad_z = slice_by_range(quad_store.z, quad_desc.z_range);
		auto quad_w = slice_by_range(quad_store.w, quad_desc.w_range);

		if (need_basis_values)
		{
			auto basis_val = slice_by_range<double>(cache.basis_values, desc.basis_val_range);
			basis::ng::eval_basis_values_at_points(
				bases,
				element_id,
				quad_num,
				quad_x,
				quad_y,
				quad_z,
				basis_val);
		}

		if (need_basis_gradients)
		{
			auto grad_x = slice_by_range<double>(cache.basis_grad_x, desc.basis_grad_x_range);
			auto grad_y = slice_by_range<double>(cache.basis_grad_y, desc.basis_grad_y_range);
			auto grad_z = slice_by_range<double>(cache.basis_grad_z, desc.basis_grad_z_range);
			basis::ng::eval_basis_gradients_at_points(
				bases,
				element_id,
				quad_num,
				quad_x,
				quad_y,
				quad_z,
				grad_x,
				grad_y,
				grad_z);
		}

		span<const double> geom_basis_values;
		span<const double> geom_basis_grad_x;
		span<const double> geom_basis_grad_y;
		span<const double> geom_basis_grad_z;
		// Isoparametric (geom basis == basis).
		// Point geom basis span to basis.
		if (is_isoparametric)
		{
			geom_basis_values = slice_by_range<double>(cache.basis_values, desc.basis_val_range);
			geom_basis_grad_x = slice_by_range<double>(cache.basis_grad_x, desc.basis_grad_x_range);
			geom_basis_grad_y = slice_by_range<double>(cache.basis_grad_y, desc.basis_grad_y_range);
			geom_basis_grad_z = slice_by_range<double>(cache.basis_grad_z, desc.basis_grad_z_range);
		}
		// Non isoparamtric. Eval geometry basis value and gradient into geom_mapping_scratch.
		else
		{
			basis::ng::eval_basis_values_at_points(
				geom_bases,
				element_id,
				quad_num,
				quad_x,
				quad_y,
				quad_z,
				geom_mapping_scratch.basis_values);
			geom_basis_values = geom_mapping_scratch.basis_values;

			basis::ng::eval_basis_gradients_at_points(
				geom_bases,
				element_id,
				quad_num,
				quad_x,
				quad_y,
				quad_z,
				geom_mapping_scratch.basis_grad_x,
				geom_mapping_scratch.basis_grad_y,
				geom_mapping_scratch.basis_grad_z);
			geom_basis_grad_x = geom_mapping_scratch.basis_grad_x;
			geom_basis_grad_y = geom_mapping_scratch.basis_grad_y;
			geom_basis_grad_z = geom_mapping_scratch.basis_grad_z;
		}

		auto phy_x = slice_by_range<double>(cache.physical_x, desc.physical_x_range);
		auto phy_y = slice_by_range<double>(cache.physical_y, desc.physical_y_range);
		auto phy_z = slice_by_range<double>(cache.physical_z, desc.physical_z_range);
		auto det_J = slice_by_range<double>(cache.det_J, desc.det_J_range);
		auto J_it = slice_by_range<double>(cache.J_inverse_transpose, desc.J_inverse_transpose_range);
		basis::ng::eval_geometry_mapping(
			geom_bases,
			element_id,
			quad_num,
			geom_basis_values,
			geom_basis_grad_x,
			geom_basis_grad_y,
			geom_basis_grad_z,
			phy_x,
			phy_y,
			phy_z,
			det_J,
			J_it);

		auto weighted_measure = slice_by_range<double>(cache.weighted_measure, desc.weighted_measure_range);
		compute_weighted_measure(quad_w, det_J, weighted_measure);

		if (need_basis_gradients)
		{
			auto basis_grad_x = slice_by_range<double>(cache.basis_grad_x, desc.basis_grad_x_range);
			auto basis_grad_y = slice_by_range<double>(cache.basis_grad_y, desc.basis_grad_y_range);
			auto basis_grad_z = slice_by_range<double>(cache.basis_grad_z, desc.basis_grad_z_range);
			auto basis_grad_phy_x = slice_by_range<double>(cache.basis_grad_phy_x, desc.basis_grad_phy_x_range);
			auto basis_grad_phy_y = slice_by_range<double>(cache.basis_grad_phy_y, desc.basis_grad_phy_y_range);
			auto basis_grad_phy_z = slice_by_range<double>(cache.basis_grad_phy_z, desc.basis_grad_phy_z_range);
			compute_basis_grad_phy(
				dim,
				basis_num,
				quad_num,
				basis_grad_x,
				basis_grad_y,
				basis_grad_z,
				J_it,
				basis_grad_phy_x,
				basis_grad_phy_y,
				basis_grad_phy_z);
		}
	}

	AssemblyCache compute_assembly_cache_batched(
		const basis::ng::ElementBasesView &bases,
		const basis::ng::ElementBasesView &geom_bases,
		bool is_mass,
		bool is_isoparametric,
		bool need_basis_values,
		bool need_basis_gradients)
	{
		GeomMappingScratch geom_scratch;
		AssemblyCache cache;
		for (int e = 0; e < bases.element_desc.size(); ++e)
		{
			// TODO: smart caching
			// - For low order basis, dont cache.
			// - Based on user json flag and element size.

			compute_assembly_cache_single(
				bases,
				geom_bases,
				e,
				is_mass,
				is_isoparametric,
				need_basis_values,
				need_basis_gradients,
				geom_scratch,
				cache);
		}
		return cache;
	}

} // namespace polyfem::assembler
