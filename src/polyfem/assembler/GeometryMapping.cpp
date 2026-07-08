#include <polyfem/assembler/GeometryMapping.hpp>

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
				vec(1) = y[id];
			if constexpr (dim > 2)
				vec(2) = z[id];
			return vec;
		}

		template <int dim>
		void vec2xyz(int id, const Eigen::Vector<double, dim> &vec, Span<double> x, Span<double> y, Span<double> z)
		{
			x[id] = vec(0);
			if constexpr (dim > 1)
				y[id] = vec(1);
			if constexpr (dim > 2)
				z[id] = vec(2);
		}

		template <int dim>
		Eigen::Vector<double, dim> basis_node_position(
			const ElementBasesView &geom_bases,
			const ElementDesc &geom_elem_desc,
			int local_basis_id)
		{
			const int mapping_id = geom_elem_desc.dof_mapping_range.offset + local_basis_id;
			const auto node_ids = geom_bases.dof_mapping_store.get_node_ids(mapping_id);
			const auto weights = geom_bases.dof_mapping_store.get_weights(mapping_id);
			const auto node_positions = geom_bases.dof_mapping_store.get_positions(mapping_id);

			Eigen::Vector<double, dim> result = Eigen::Vector<double, dim>::Zero();
			for (int i = 0; i < node_ids.size(); ++i)
				result += weights[i] * Eigen::Map<const Eigen::Vector<double, dim>>(node_positions.data() + i * dim);
			return result;
		}

		template <int dim>
		void assert_position_sizes(
			int quad_num,
			Span<const double> y,
			Span<const double> z,
			Span<double> physical_x,
			Span<double> physical_y,
			Span<double> physical_z)
		{
			assert(physical_x.size() == quad_num);
			if constexpr (dim > 1)
			{
				assert(y.size() == quad_num);
				assert(physical_y.size() == quad_num);
			}
			else
			{
				assert(y.empty());
				assert(physical_y.empty());
			}

			if constexpr (dim > 2)
			{
				assert(z.size() == quad_num);
				assert(physical_z.size() == quad_num);
			}
			else
			{
				assert(z.empty());
				assert(physical_z.empty());
			}
		}
	} // namespace

	template <int dim>
	void compute_geometry_positions(
		const ElementBasesView &geom_bases,
		int element_id,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> physical_x,
		Span<double> physical_y,
		Span<double> physical_z,
		Span<double> geom_basis_values)
	{
		assert(0 <= element_id && element_id < geom_bases.element_desc.size());
		auto &geom_elem_desc = geom_bases.element_desc[element_id];
		int quad_num = x.size();
		int geom_basis_num = geom_elem_desc.basis_desc.basis_num;

		assert_position_sizes<dim>(quad_num, y, z, physical_x, physical_y, physical_z);

		// For non parametric basis, quadrature point is already physical position.
		if (!geom_elem_desc.basis_desc.is_parametric)
		{
			std::copy(x.begin(), x.end(), physical_x.begin());
			if constexpr (dim > 1)
				std::copy(y.begin(), y.end(), physical_y.begin());
			if constexpr (dim > 2)
				std::copy(z.begin(), z.end(), physical_z.begin());
			return;
		}

		assert(geom_basis_values.size() == geom_basis_num * quad_num);
		basis::basis_values(
			geom_elem_desc.basis_desc,
			geom_bases.basis_store,
			x,
			y,
			z,
			geom_basis_values);

		using Vec = Eigen::Vector<double, dim>;
		for (int q = 0; q < quad_num; ++q)
		{
			Vec phy_pos = Vec::Zero();
			for (int b = 0; b < geom_basis_num; ++b)
			{
				phy_pos += geom_basis_values[b * quad_num + q] * basis_node_position<dim>(geom_bases, geom_elem_desc, b);
			}
			vec2xyz<dim>(q, phy_pos, physical_x, physical_y, physical_z);
		}
	}

	template <int dim>
	void compute_geometry_mapping(
		const ElementBasesView &geom_bases,
		int element_id,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> physical_x,
		Span<double> physical_y,
		Span<double> physical_z,
		Span<double> det_J,
		Span<double> J_inverse_transpose,
		Span<double> geom_basis_values,
		Span<double> geom_basis_grad_x,
		Span<double> geom_basis_grad_y,
		Span<double> geom_basis_grad_z)
	{
		assert(0 <= element_id && element_id < geom_bases.element_desc.size());
		auto &geom_elem_desc = geom_bases.element_desc[element_id];
		int quad_num = x.size();
		int geom_basis_num = geom_elem_desc.basis_desc.basis_num;

		assert_position_sizes<dim>(quad_num, y, z, physical_x, physical_y, physical_z);
		assert(det_J.size() == quad_num);
		assert(J_inverse_transpose.size() == quad_num * dim * dim);

		using Vec = Eigen::Vector<double, dim>;
		using Mat = Eigen::Matrix<double, dim, dim, Eigen::RowMajor>;

		if (!geom_elem_desc.basis_desc.is_parametric)
		{
			std::copy(x.begin(), x.end(), physical_x.begin());
			if constexpr (dim > 1)
				std::copy(y.begin(), y.end(), physical_y.begin());
			if constexpr (dim > 2)
				std::copy(z.begin(), z.end(), physical_z.begin());

			for (int q = 0; q < quad_num; ++q)
			{
				Mat J_it = Mat::Identity();
				det_J[q] = 1.0;
				std::copy(J_it.data(), J_it.data() + J_it.size(), J_inverse_transpose.data() + dim * dim * q);
			}
			return;
		}

		assert(geom_basis_values.size() == geom_basis_num * quad_num);
		assert(geom_basis_grad_x.size() == geom_basis_num * quad_num);
		if constexpr (dim > 1)
			assert(geom_basis_grad_y.size() == geom_basis_num * quad_num);
		else
			assert(geom_basis_grad_y.empty());
		if constexpr (dim > 2)
			assert(geom_basis_grad_z.size() == geom_basis_num * quad_num);
		else
			assert(geom_basis_grad_z.empty());

		basis::basis_value_and_gradients(
			geom_elem_desc.basis_desc,
			geom_bases.basis_store,
			x,
			y,
			z,
			geom_basis_values,
			geom_basis_grad_x,
			geom_basis_grad_y,
			geom_basis_grad_z);

		for (int q = 0; q < quad_num; ++q)
		{
			Vec phy_pos = Vec::Zero();
			Mat J = Mat::Zero();
			for (int b = 0; b < geom_basis_num; ++b)
			{
				Vec basis_node_pos = basis_node_position<dim>(geom_bases, geom_elem_desc, b);
				phy_pos += geom_basis_values[b * quad_num + q] * basis_node_pos;
				Vec grad = xyz2vec<dim>(b * quad_num + q, geom_basis_grad_x, geom_basis_grad_y, geom_basis_grad_z);
				J += grad * basis_node_pos.transpose();
			}

			const Mat J_it = J.inverse().transpose();
			vec2xyz<dim>(q, phy_pos, physical_x, physical_y, physical_z);
			det_J[q] = J.determinant();
			std::copy(J_it.data(), J_it.data() + J_it.size(), J_inverse_transpose.data() + dim * dim * q);
		}
	}

	template void compute_geometry_positions<1>(const ElementBasesView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>);
	template void compute_geometry_positions<2>(const ElementBasesView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>);
	template void compute_geometry_positions<3>(const ElementBasesView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>);

	template void compute_geometry_mapping<1>(const ElementBasesView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>);
	template void compute_geometry_mapping<2>(const ElementBasesView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>);
	template void compute_geometry_mapping<3>(const ElementBasesView &, int, Span<const double>, Span<const double>, Span<const double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>, Span<double>);

} // namespace polyfem::assembler
