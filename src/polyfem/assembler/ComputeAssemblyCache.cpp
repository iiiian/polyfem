#include <polyfem/assembler/ComputeAssemblyCache.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>
#include <polyfem/basis/EvalBasis.hpp>

#include <algorithm>
#include <cassert>
#include <Eigen/Core>

namespace polyfem::assembler
{
	namespace
	{
		template <int dim>
		Eigen::Vector<double, dim> xyz2vec(int id, Span<const double> x, Span<const double> y, Span<const double> z)
		{
			Eigen::Vector<double, dim> vec = Eigen::Vector<double, dim>::Zero();
			vec(0) = x[id];
			if constexpr (dim > 1)
			{
				vec(1) = y[id];
			}
			if constexpr (dim > 2)
			{
				vec(2) = z[id];
			}

			return vec;
		}

		template <int dim>
		void vec2xyz(int id, Eigen::Vector<double, dim> vec, Span<double> x, Span<double> y, Span<double> z)
		{
			x[id] = vec(0);
			if constexpr (dim > 1)
			{
				y[id] = vec(1);
			}
			if constexpr (dim > 2)
			{
				z[id] = vec(2);
			}
		}

		template <int dim>
		void compute_geom_basis_and_mapping(
			const ElementBasesView &bases,
			const ElementBasesView &geom_bases,
			int element_id,
			bool is_mass,
			AssemblyTempStorage temp)
		{
			auto &elem_desc = bases.element_desc[element_id];
			auto &geom_elem_desc = geom_bases.element_desc[element_id];

			// Quadrature points and weights.
			auto &quad_desc = is_mass ? elem_desc.mass_quadrature_desc : elem_desc.quadrature_desc;
			auto &quad_store = is_mass ? bases.mass_quadrature : bases.quadrature;
			int quad_num = quad_desc.w_range.num;
			auto quad_x = quad_store.get_x(quad_desc);
			auto quad_y = quad_store.get_y(quad_desc);
			auto quad_z = quad_store.get_z(quad_desc);
			auto quad_w = quad_store.get_w(quad_desc);

			int geom_basis_num = geom_elem_desc.basis_desc.basis_num;

			basis::basis_value_and_gradients(
				geom_elem_desc.basis_desc,
				geom_bases.basis,
				quad_x,
				quad_y,
				quad_z,
				temp.gbasis_values,
				temp.gbasis_grad_x,
				temp.gbasis_grad_y,
				temp.gbasis_grad_z);

			using Vec = Eigen::Vector<double, dim>;
			using Mat = Eigen::Matrix<double, dim, dim, Eigen::RowMajor>;

			for (int qi = 0; qi < quad_num; ++qi)
			{
				Vec phy_pos = Vec::Zero();
				Mat J = Mat::Zero();
				for (int bi = 0; bi < geom_basis_num; ++bi)
				{
					auto &mappings = geom_bases.dof_mapping;
					int mapping_id = geom_elem_desc.dof_mapping_range.offset + bi;
					auto node_ids = mappings.get_node_ids(mapping_id);
					auto weights = mappings.get_weights(mapping_id);
					auto node_positions = mappings.get_positions(mapping_id);

					Vec basis_node_pos = Vec::Zero();

					for (int mi = 0; mi < node_ids.size(); ++mi)
					{
						basis_node_pos += weights[mi] * Eigen::Map<const Vec>(node_positions.data() + mi * dim);
					}

					phy_pos += temp.gbasis_values[bi * quad_num + qi] * basis_node_pos;
					Vec grad = xyz2vec<dim>(bi * quad_num + qi, temp.gbasis_grad_x, temp.gbasis_grad_y, temp.gbasis_grad_z);
					J += grad * basis_node_pos.transpose();
				}

				double det_J = J.determinant();
				Mat J_it = J.inverse().transpose();

				// write output to temp.
				vec2xyz(qi, phy_pos, temp.physical_x, temp.physical_y, temp.physical_z);
				temp.det_J[qi] = det_J;
				temp.weighted_measure[qi] = det_J * quad_w[qi];
				std::copy(J_it.data(), J_it.data() + J_it.size(), temp.J_inverse_transpose.data() + dim * dim * qi);
			}
		}

		template <int dim>
		void fill_identity_geom_mapping(
			const ElementBasesView &bases,
			const ElementBasesView &geom_bases,
			int element_id,
			bool is_mass,
			AssemblyTempStorage temp)
		{
			auto &elem_desc = bases.element_desc[element_id];
			auto &geom_elem_desc = geom_bases.element_desc[element_id];

			// Quadrature points and weights.
			auto &quad_desc = is_mass ? elem_desc.mass_quadrature_desc : elem_desc.quadrature_desc;
			auto &quad_store = is_mass ? bases.mass_quadrature : bases.quadrature;
			int quad_num = quad_desc.w_range.num;
			auto quad_x = quad_store.get_x(quad_desc);
			auto quad_y = quad_store.get_y(quad_desc);
			auto quad_z = quad_store.get_z(quad_desc);
			auto quad_w = quad_store.get_w(quad_desc);

			using Vec = Eigen::Vector<double, dim>;
			using Mat = Eigen::Matrix<double, dim, dim, Eigen::RowMajor>;

			for (int qi = 0; qi < quad_num; ++qi)
			{
				Vec phy_pos = xyz2vec<dim>(qi, quad_x, quad_y, quad_z);
				Mat J_it = Mat::Identity();
				double det_J = 1.0;

				// write output to temp.
				vec2xyz(qi, phy_pos, temp.physical_x, temp.physical_y, temp.physical_z);
				temp.det_J[qi] = det_J;
				temp.weighted_measure[qi] = quad_w[qi];
				std::copy(J_it.data(), J_it.data() + J_it.size(), temp.J_inverse_transpose.data() + dim * dim * qi);
			}
		}

	} // namespace

	template <int dim>
	void compute_assembly_cache_single(
		const ElementBasesView &bases,
		const ElementBasesView &geom_bases,
		int element_id,
		bool is_mass,
		AssemblyTempStorage temp)
	{
		assert(0 <= element_id && element_id < bases.element_desc.size());
		assert(0 <= element_id && element_id < geom_bases.element_desc.size());

		auto &elem_desc = bases.element_desc[element_id];
		auto &geom_elem_desc = geom_bases.element_desc[element_id];
		auto &quad_desc = is_mass ? elem_desc.mass_quadrature_desc : elem_desc.quadrature_desc;

		temp.resize(dim, elem_desc.basis_desc.basis_num, geom_elem_desc.basis_desc.basis_num, quad_desc.w_range.num);

		// Compute geometry mappings.

		if (geom_elem_desc.basis_desc.is_parametric)
		{
			compute_geom_basis_and_mapping<dim>(bases, geom_bases, element_id, is_mass, temp);
		}
		else
		{
			fill_identity_geom_mapping<dim>(bases, geom_bases, element_id, is_mass, temp);
		}

		// Quadrature points and weights.
		int quad_num = quad_desc.w_range.num;
		auto &quad_store = is_mass ? bases.mass_quadrature : bases.quadrature;
		auto quad_x = quad_store.get_x(quad_desc);
		auto quad_y = quad_store.get_y(quad_desc);
		auto quad_z = quad_store.get_z(quad_desc);
		auto quad_w = quad_store.get_w(quad_desc);

		// Compute basis values and gradients.
		int basis_num = elem_desc.basis_desc.basis_num;
		basis::basis_value_and_gradients(
			elem_desc.basis_desc,
			bases.basis,
			quad_x,
			quad_y,
			quad_z,
			temp.basis_values,
			temp.basis_grad_x,
			temp.basis_grad_y,
			temp.basis_grad_z);

		// Compute basis grad physical.
		using Vec = Eigen::Vector<double, dim>;
		using Mat = Eigen::Matrix<double, dim, dim, Eigen::RowMajor>;
		for (int bi = 0; bi < basis_num; ++bi)
		{
			for (int qi = 0; qi < quad_num; ++qi)
			{
				Vec grad = xyz2vec<dim>(bi * quad_num + qi, temp.basis_grad_x, temp.basis_grad_y, temp.basis_grad_z);
				auto J_it = Eigen::Map<Mat>(temp.J_inverse_transpose.data() + dim * dim * qi);
				Vec grad_phy = J_it * grad;
				vec2xyz<dim>(bi * quad_num + qi, grad_phy, temp.basis_grad_phy_x, temp.basis_grad_phy_y, temp.basis_grad_phy_z);
			}
		}
	}

	template void compute_assembly_cache_single<1>(const ElementBasesView &, const ElementBasesView &, int, bool, AssemblyTempStorage);
	template void compute_assembly_cache_single<2>(const ElementBasesView &, const ElementBasesView &, int, bool, AssemblyTempStorage);
	template void compute_assembly_cache_single<3>(const ElementBasesView &, const ElementBasesView &, int, bool, AssemblyTempStorage);

	AssemblyCache compute_assembly_cache_batched(
		const ElementBasesView &bases,
		const ElementBasesView &geom_bases,
		bool is_mass)
	{
		AssemblyTempStorage temp;
		AssemblyCache cache;
		for (int e = 0; e < bases.element_desc.size(); ++e)
		{
			// TODO: smart caching
			// - For low order basis, dont cache?
			// - Based on user json flag and element size.

			auto &elem_desc = bases.element_desc[e];
			int dim = elem_desc.basis_desc.dim;

			switch (dim)
			{
			case 1:
				compute_assembly_cache_single<1>(bases, geom_bases, e, is_mass, temp);
				break;
			case 2:
				compute_assembly_cache_single<2>(bases, geom_bases, e, is_mass, temp);
				break;
			case 3:
				compute_assembly_cache_single<3>(bases, geom_bases, e, is_mass, temp);
				break;
			default:
				assert(false);
			}

			cache.append(temp);
		}
		return cache;
	}

} // namespace polyfem::assembler
