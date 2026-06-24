#include <polyfem/basis/ElementBases.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_pyramid_bases.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>
#include <polyfem/autogen/prism_bases.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>

namespace polyfem::basis::ng
{
	using namespace assembler;
	namespace
	{
		using RowMatrix2d = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
		using RowMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

		void fill_points(
			const BasisDesc &basis_desc,
			int n_points,
			span<const double> point_x,
			span<const double> point_y,
			span<const double> point_z,
			Eigen::MatrixXd &uv)
		{
			int dim = basis_desc.dim;

			uv.resize(n_points, dim);
			for (int q = 0; q < n_points; ++q)
			{
				uv(q, 0) = point_x[q];
				if (dim >= 2)
					uv(q, 1) = point_y[q];
				if (dim >= 3)
					uv(q, 2) = point_z[q];
			}
		}

		void eval_lagrange_basis_value(
			const BasisDesc &basis_desc,
			int local_i,
			const Eigen::MatrixXd &uv,
			Eigen::MatrixXd &val)
		{
			assert(basis_desc.basis_family == BasisFamily::Lagrange);

			if (basis_desc.element_kind == ElementKind::Simplex)
			{
				if (basis_desc.dim == 2)
					autogen::p_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, local_i, uv, val);
				else
					autogen::p_basis_value_3d(basis_desc.is_bernstein, basis_desc.order, local_i, uv, val);
			}
			else if (basis_desc.element_kind == ElementKind::Quad)
			{
				assert(basis_desc.dim == 2);
				autogen::q_basis_value_2d(basis_desc.order, local_i, uv, val);
			}
			else if (basis_desc.element_kind == ElementKind::Hex)
			{
				assert(basis_desc.dim == 3);
				autogen::q_basis_value_3d(basis_desc.order, local_i, uv, val);
			}
			else if (basis_desc.element_kind == ElementKind::Prism)
			{
				assert(basis_desc.dim == 3);
				autogen::prism_basis_value_3d(basis_desc.order, basis_desc.orderq, local_i, uv, val);
			}
			else if (basis_desc.element_kind == ElementKind::Pyramid)
			{
				assert(basis_desc.dim == 3);
				autogen::pyramid_basis_value_3d(basis_desc.order, local_i, uv, val);
			}
			else
			{
				assert(false);
			}
		}

		void eval_lagrange_basis_gradient(
			const BasisDesc &basis_desc,
			int local_i,
			const Eigen::MatrixXd &uv,
			Eigen::MatrixXd &grad)
		{
			assert(basis_desc.basis_family == BasisFamily::Lagrange);

			if (basis_desc.element_kind == ElementKind::Simplex)
			{
				if (basis_desc.dim == 2)
					autogen::p_grad_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, local_i, uv, grad);
				else
					autogen::p_grad_basis_value_3d(basis_desc.is_bernstein, basis_desc.order, local_i, uv, grad);
			}
			else if (basis_desc.element_kind == ElementKind::Quad)
			{
				assert(basis_desc.dim == 2);
				autogen::q_grad_basis_value_2d(basis_desc.order, local_i, uv, grad);
			}
			else if (basis_desc.element_kind == ElementKind::Hex)
			{
				assert(basis_desc.dim == 3);
				autogen::q_grad_basis_value_3d(basis_desc.order, local_i, uv, grad);
			}
			else if (basis_desc.element_kind == ElementKind::Prism)
			{
				assert(basis_desc.dim == 3);
				autogen::prism_grad_basis_value_3d(basis_desc.order, basis_desc.orderq, local_i, uv, grad);
			}
			else if (basis_desc.element_kind == ElementKind::Pyramid)
			{
				assert(basis_desc.dim == 3);
				autogen::pyramid_grad_basis_value_3d(basis_desc.order, local_i, uv, grad);
			}
			else
			{
				assert(false);
			}
		}

		span<const double> rational_weights(const ElementBasesView &bases, const BasisDesc &basis_desc)
		{
			assert(basis_desc.basis_family == BasisFamily::Rational);
			auto weights = slice_by_range(bases.basis.rational_weights, basis_desc.rational_weight_range);
			assert(weights.size() == local_basis_num(basis_desc));
			return weights;
		}

		void eval_rational_basis_value(
			const ElementBasesView &bases,
			const BasisDesc &basis_desc,
			int local_i,
			const Eigen::MatrixXd &uv,
			Eigen::MatrixXd &val)
		{
			assert(basis_desc.element_kind == ElementKind::Simplex);
			assert(basis_desc.dim == 2);

			int n_bases = local_basis_num(basis_desc);
			span<const double> weights = rational_weights(bases, basis_desc);

			autogen::p_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, local_i, uv, val);

			Eigen::MatrixXd denom = Eigen::MatrixXd::Zero(val.rows(), val.cols());
			Eigen::MatrixXd tmp;
			for (int k = 0; k < n_bases; ++k)
			{
				autogen::p_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, k, uv, tmp);
				denom += weights[k] * tmp;
			}

			val = (weights[local_i] * val.array() / denom.array()).eval();
		}

		void eval_rational_basis_gradient(
			const ElementBasesView &bases,
			const BasisDesc &basis_desc,
			int local_i,
			const Eigen::MatrixXd &uv,
			Eigen::MatrixXd &grad)
		{
			assert(basis_desc.element_kind == ElementKind::Simplex);
			assert(basis_desc.dim == 2);

			int n_bases = local_basis_num(basis_desc);
			span<const double> weights = rational_weights(bases, basis_desc);

			Eigen::MatrixXd basis_val;
			autogen::p_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, local_i, uv, basis_val);
			autogen::p_grad_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, local_i, uv, grad);

			Eigen::MatrixXd denom = Eigen::MatrixXd::Zero(basis_val.rows(), basis_val.cols());
			Eigen::MatrixXd denom_grad = Eigen::MatrixXd::Zero(grad.rows(), grad.cols());
			Eigen::MatrixXd tmp;

			for (int k = 0; k < n_bases; ++k)
			{
				autogen::p_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, k, uv, tmp);
				denom += weights[k] * tmp;

				autogen::p_grad_basis_value_2d(basis_desc.is_bernstein, basis_desc.order, k, uv, tmp);
				denom_grad += weights[k] * tmp;
			}

			Eigen::ArrayXd denom_value = denom.col(0).array();
			Eigen::ArrayXd basis_value = basis_val.col(0).array();
			Eigen::ArrayXd denom_sq = denom_value * denom_value;
			grad.col(0) = ((weights[local_i] * grad.col(0).array() * denom_value - weights[local_i] * basis_value * denom_grad.col(0).array()) / denom_sq).eval();
			grad.col(1) = ((weights[local_i] * grad.col(1).array() * denom_value - weights[local_i] * basis_value * denom_grad.col(1).array()) / denom_sq).eval();
		}

		void eval_basis_value(
			const ElementBasesView &bases,
			const BasisDesc &basis_desc,
			int local_i,
			const Eigen::MatrixXd &uv,
			Eigen::MatrixXd &val)
		{
			if (basis_desc.basis_family == BasisFamily::Lagrange)
				eval_lagrange_basis_value(basis_desc, local_i, uv, val);
			else if (basis_desc.basis_family == BasisFamily::Rational)
				eval_rational_basis_value(bases, basis_desc, local_i, uv, val);
			else
				assert(false);
		}

		void eval_basis_gradient(
			const ElementBasesView &bases,
			const BasisDesc &basis_desc,
			int local_i,
			const Eigen::MatrixXd &uv,
			Eigen::MatrixXd &grad)
		{
			if (basis_desc.basis_family == BasisFamily::Lagrange)
				eval_lagrange_basis_gradient(basis_desc, local_i, uv, grad);
			else if (basis_desc.basis_family == BasisFamily::Rational)
				eval_rational_basis_gradient(bases, basis_desc, local_i, uv, grad);
			else
				assert(false);
		}
	} // namespace

	ElementBasesView ElementBases::view() const
	{
		return ElementBasesView{
			element_desc,
			quadrature_store.view(),
			mass_quadrature_store.view(),
			dof_mapping_store.view(),
			basis_store.view()};
	}

	int local_basis_num(const BasisDesc &basis_desc)
	{
		if (basis_desc.element_kind == ElementKind::Simplex)
		{
			int p = basis_desc.order;
			if (basis_desc.dim == 2)
				return (p + 1) * (p + 2) / 2;
			else if (basis_desc.dim == 3)
				return (p + 1) * (p + 2) * (p + 3) / 6;
		}
		else if (basis_desc.element_kind == ElementKind::Quad)
		{
			assert(basis_desc.dim == 2);
			int q = basis_desc.order;
			if (q == -2)
				return 8;
			return (q + 1) * (q + 1);
		}
		else if (basis_desc.element_kind == ElementKind::Hex)
		{
			assert(basis_desc.dim == 3);
			int q = basis_desc.order;
			if (q == -2)
				return 20;
			return (q + 1) * (q + 1) * (q + 1);
		}
		else if (basis_desc.element_kind == ElementKind::Prism)
		{
			assert(basis_desc.dim == 3);
			int p = basis_desc.order;
			int q = basis_desc.orderq;
			return (p + 1) * (p + 2) * (q + 1) / 2;
		}
		else if (basis_desc.element_kind == ElementKind::Pyramid)
		{
			assert(basis_desc.dim == 3);
			int p = basis_desc.order;
			return (p + 1) * (p + 2) * (2 * p + 3) / 6;
		}

		assert(false);
		return 0;
	}

	void eval_basis_values_at_points(
		const ElementBasesView &bases,
		int element_id,
		int quad_num,
		span<const double> qx,
		span<const double> qy,
		span<const double> qz,
		span<double> values)
	{
		assert(0 <= element_id && element_id < bases.element_desc.size());
		assert(0 <= quad_num);

		auto basis_desc = bases.element_desc[element_id].basis_desc;
		int basis_num = local_basis_num(basis_desc);
		assert(values.size() == basis_num * quad_num);

		Eigen::MatrixXd uv;
		fill_points(basis_desc, quad_num, qx, qy, qz, uv);

		Eigen::MatrixXd tmp_out;
		for (int i = 0; i < basis_num; ++i)
		{
			eval_basis_value(bases, basis_desc, i, uv, tmp_out);
			assert(tmp_out.rows() == quad_num);
			assert(tmp_out.cols() == 1);
			std::memcpy(values.data() + i * quad_num, tmp_out.data(), quad_num * sizeof(double));
		}
	}

	void eval_basis_gradients_at_points(
		const ElementBasesView &bases,
		int element_id,
		int n_points,
		span<const double> point_x,
		span<const double> point_y,
		span<const double> point_z,
		span<double> grad_x,
		span<double> grad_y,
		span<double> grad_z)
	{
		assert(0 <= element_id && element_id < int(bases.element_desc.size()));
		assert(0 <= n_points);

		const BasisDesc &basis_desc = bases.element_desc[element_id].basis_desc;
		const int dim = basis_desc.dim;
		const int n_bases = local_basis_num(basis_desc);
		const int n_entries = n_bases * n_points;
		assert(grad_x.size() == std::size_t(n_entries));
		assert(dim < 2 || grad_y.size() == std::size_t(n_entries));
		assert(dim < 3 || grad_z.size() == std::size_t(n_entries));

		Eigen::MatrixXd uv;
		fill_points(basis_desc, n_points, point_x, point_y, point_z, uv);

		Eigen::MatrixXd grad;
		for (int local_i = 0; local_i < n_bases; ++local_i)
		{
			eval_basis_gradient(bases, basis_desc, local_i, uv, grad);
			assert(grad.rows() == n_points);
			assert(grad.cols() == dim);

			for (int q = 0; q < n_points; ++q)
			{
				int offset = local_i * n_points + q;
				grad_x[offset] = grad(q, 0);
				if (dim >= 2)
					grad_y[offset] = grad(q, 1);
				if (dim >= 3)
					grad_z[offset] = grad(q, 2);
			}
		}
	}

	void eval_basis_values_at_quadrature(
		const ElementBasesView &bases,
		int element_id,
		bool is_mass,
		span<double> values)
	{
		assert(0 <= element_id && element_id < int(bases.element_desc.size()));
		const ElementDesc &element = bases.element_desc[element_id];
		const QuadratureDesc &quad = is_mass ? element.mass_quadrature_desc : element.quadrature_desc;
		const QuadratureStoreView &store = is_mass ? bases.mass_quadrature : bases.quadrature;

		eval_basis_values_at_points(
			bases,
			element_id,
			quad.w_range.num,
			slice_by_range<const double>(store.x, quad.x_range),
			slice_by_range<const double>(store.y, quad.y_range),
			slice_by_range<const double>(store.z, quad.z_range),
			values);
	}

	void eval_basis_gradients_at_quadrature(
		const ElementBasesView &bases,
		int element_id,
		bool is_mass,
		span<double> grad_x,
		span<double> grad_y,
		span<double> grad_z)
	{
		assert(0 <= element_id && element_id < int(bases.element_desc.size()));
		const ElementDesc &element = bases.element_desc[element_id];
		const QuadratureDesc &quad = is_mass ? element.mass_quadrature_desc : element.quadrature_desc;
		const QuadratureStoreView &store = is_mass ? bases.mass_quadrature : bases.quadrature;

		eval_basis_gradients_at_points(
			bases,
			element_id,
			quad.w_range.num,
			slice_by_range<const double>(store.x, quad.x_range),
			slice_by_range<const double>(store.y, quad.y_range),
			slice_by_range<const double>(store.z, quad.z_range),
			grad_x,
			grad_y,
			grad_z);
	}

	void eval_geometry_mapping(
		const ElementBasesView &geometry_bases,
		int element_id,
		int n_points,
		span<const double> basis_values,
		span<const double> basis_grad_x,
		span<const double> basis_grad_y,
		span<const double> basis_grad_z,
		span<double> mapped_x,
		span<double> mapped_y,
		span<double> mapped_z,
		span<double> det_jacobian,
		span<double> jacobian_inverse_transpose)
	{
		assert(0 <= element_id && element_id < int(geometry_bases.element_desc.size()));
		assert(0 <= n_points);

		auto &element = geometry_bases.element_desc[element_id];
		auto &basis_desc = element.basis_desc;
		int dim = basis_desc.dim;
		int n_bases = local_basis_num(basis_desc);
		int n_entries = n_bases * n_points;
		int jac_entries = n_points * dim * dim;

		std::array<span<double>, 3> mapped = {{mapped_x, mapped_y, mapped_z}};
		std::array<span<const double>, 3> basis_grad = {{basis_grad_x, basis_grad_y, basis_grad_z}};

		for (int d = 0; d < dim; ++d)
			std::fill(mapped[d].begin(), mapped[d].end(), 0);
		std::fill(jacobian_inverse_transpose.begin(), jacobian_inverse_transpose.end(), 0);

		for (int local_i = 0; local_i < n_bases; ++local_i)
		{
			int mapping_id = element.dof_mapping_range.offset + local_i;
			assert(0 <= mapping_id && mapping_id < int(geometry_bases.dof_mapping.mapping_desc.size()));
			auto &mapping = geometry_bases.dof_mapping.mapping_desc[mapping_id];

			int mapping_count = mapping.id_and_weight_range.num;

			for (int mapping_i = 0; mapping_i < mapping_count; ++mapping_i)
			{
				int value_id = mapping.id_and_weight_range.offset + mapping_i;
				int node_id = mapping.node_position_range.offset + mapping_i * dim;
				double weight = geometry_bases.dof_mapping.weights[value_id];
				std::array<double, 3> node = {{0, 0, 0}};
				for (int d = 0; d < dim; ++d)
					node[d] = weight * geometry_bases.dof_mapping.node_positions[node_id + d];

				for (int q = 0; q < n_points; ++q)
				{
					int entry = local_i * n_points + q;
					int jac_offset = q * dim * dim;
					double value = basis_values[entry];

					for (int d = 0; d < dim; ++d)
						mapped[d][q] += value * node[d];

					for (int r = 0; r < dim; ++r)
					{
						double grad = basis_grad[r][entry];
						for (int d = 0; d < dim; ++d)
							jacobian_inverse_transpose[jac_offset + r * dim + d] += grad * node[d];
					}
				}
			}
		}

		if (dim == 1)
		{
			for (int q = 0; q < n_points; ++q)
			{
				double &jac = jacobian_inverse_transpose[q];
				det_jacobian[q] = jac;
				jac = 1.0 / jac;
			}
		}
		else if (dim == 2)
		{
			for (int q = 0; q < n_points; ++q)
			{
				Eigen::Map<RowMatrix2d> jac(jacobian_inverse_transpose.data() + q * dim * dim);
				det_jacobian[q] = jac.determinant();
				RowMatrix2d inverse_transpose = jac.inverse().transpose();
				jac = inverse_transpose;
			}
		}
		else
		{
			for (int q = 0; q < n_points; ++q)
			{
				Eigen::Map<RowMatrix3d> jac(jacobian_inverse_transpose.data() + q * dim * dim);
				det_jacobian[q] = jac.determinant();
				RowMatrix3d inverse_transpose = jac.inverse().transpose();
				jac = inverse_transpose;
			}
		}
	}

} // namespace polyfem::basis::ng
