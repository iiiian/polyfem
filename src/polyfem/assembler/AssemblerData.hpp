#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/span.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>

#include <Eigen/Core>

#include <cassert>

namespace polyfem::assembler
{
	class NonLinearElementAssemblyData
	{
	public:
		NonLinearElementAssemblyData() = default;

		NonLinearElementAssemblyData(
			int element_id,
			int cache_desc_id,
			int dim,
			int basis_num,
			double time,
			double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const basis::ng::ElementBasesView &bases,
			const AssemblyCacheView &cache)
			: element_id(element_id),
			  elem_dim(dim),
			  basis_num(basis_num),
			  t(time),
			  dt(dt),
			  x(x),
			  x_prev(x_prev),
			  element_desc(bases.element_desc[element_id]),
			  dof_mapping(bases.dof_mapping)
		{
			const AssemblyCacheDesc &desc = cache.desc[cache_desc_id];
			weighted_measure = slice_by_range(cache.weighted_measure, desc.weighted_measure_range);
			basis_values = slice_by_range(cache.basis_values, desc.basis_val_range);
			basis_grad_x = slice_by_range(cache.basis_grad_x, desc.basis_grad_x_range);
			basis_grad_y = slice_by_range(cache.basis_grad_y, desc.basis_grad_y_range);
			basis_grad_z = slice_by_range(cache.basis_grad_z, desc.basis_grad_z_range);
			basis_grad_phy_x = slice_by_range(cache.basis_grad_phy_x, desc.basis_grad_phy_x_range);
			basis_grad_phy_y = slice_by_range(cache.basis_grad_phy_y, desc.basis_grad_phy_y_range);
			basis_grad_phy_z = slice_by_range(cache.basis_grad_phy_z, desc.basis_grad_phy_z_range);
			physical_x = slice_by_range(cache.physical_x, desc.physical_x_range);
			physical_y = slice_by_range(cache.physical_y, desc.physical_y_range);
			physical_z = slice_by_range(cache.physical_z, desc.physical_z_range);
			det_J = slice_by_range(cache.det_J, desc.det_J_range);
			J_inverse_transpose = slice_by_range(cache.J_inverse_transpose, desc.J_inverse_transpose_range);

			const auto &quad_desc = desc.is_mass ? element_desc.mass_quadrature_desc : element_desc.quadrature_desc;
			const auto &quad_store = desc.is_mass ? bases.mass_quadrature : bases.quadrature;
			quadrature_x = slice_by_range(quad_store.x, quad_desc.x_range);
			quadrature_y = slice_by_range(quad_store.y, quad_desc.y_range);
			quadrature_z = slice_by_range(quad_store.z, quad_desc.z_range);
			quadrature_weight = slice_by_range(quad_store.w, quad_desc.w_range);

			quad_num = weighted_measure.size();
		}

		int element_id = -1;
		int elem_dim = 0;
		int basis_num = 0;
		int quad_num = 0;
		double t = 0.0;
		double dt = 0.0;
		const Eigen::MatrixXd &x = empty_matrix();
		const Eigen::MatrixXd &x_prev = empty_matrix();

		basis::ng::ElementDesc element_desc;
		basis::ng::DofMappingStoreView dof_mapping;

		span<const double> weighted_measure;
		span<const double> basis_values;
		span<const double> basis_grad_x;
		span<const double> basis_grad_y;
		span<const double> basis_grad_z;
		span<const double> basis_grad_phy_x;
		span<const double> basis_grad_phy_y;
		span<const double> basis_grad_phy_z;
		span<const double> physical_x;
		span<const double> physical_y;
		span<const double> physical_z;
		span<const double> det_J;
		span<const double> J_inverse_transpose;
		span<const double> quadrature_x;
		span<const double> quadrature_y;
		span<const double> quadrature_z;
		span<const double> quadrature_weight;

		double gather_basis_value(int local_basis_id, int quadrature_point_id) const
		{
			return basis_values[local_basis_id * quad_num + quadrature_point_id];
		}

		template <int dim>
		Eigen::Matrix<double, 1, dim> gather_basis_grad(int local_basis_id, int quadrature_point_id) const
		{
			const int offset = local_basis_id * quad_num + quadrature_point_id;
			return gather_row_vector<dim>(offset, basis_grad_x, basis_grad_y, basis_grad_z);
		}

		template <int dim>
		Eigen::Matrix<double, 1, dim> gather_basis_grad_phy(int local_basis_id, int quadrature_point_id) const
		{
			const int offset = local_basis_id * quad_num + quadrature_point_id;
			return gather_row_vector<dim>(offset, basis_grad_phy_x, basis_grad_phy_y, basis_grad_phy_z);
		}

		template <int dim>
		Eigen::Matrix<double, 1, dim> gather_quadrature_point(int quadrature_point_id) const
		{
			return gather_row_vector<dim>(quadrature_point_id, quadrature_x, quadrature_y, quadrature_z);
		}

		template <int dim>
		Eigen::Matrix<double, 1, dim> gather_physical_position(int quadrature_point_id) const
		{
			return gather_row_vector<dim>(quadrature_point_id, physical_x, physical_y, physical_z);
		}

		RowVectorNd quadrature_point(int quadrature_point_id) const
		{
			return gather_dynamic_row_vector(quadrature_point_id, quadrature_x, quadrature_y, quadrature_z);
		}

		RowVectorNd physical_position(int quadrature_point_id) const
		{
			return gather_dynamic_row_vector(quadrature_point_id, physical_x, physical_y, physical_z);
		}

		template <int dim>
		Eigen::Matrix<double, dim, dim, Eigen::RowMajor> gather_J_inverse_transpose(int quadrature_point_id) const
		{
			const double *ptr = J_inverse_transpose.data() + quadrature_point_id * dim * dim;
			return Eigen::Map<const Eigen::Matrix<double, dim, dim, Eigen::RowMajor>>(ptr);
		}

		Eigen::MatrixXd jacobian_inverse_transpose(int quadrature_point_id) const
		{
			const double *ptr = J_inverse_transpose.data() + quadrature_point_id * elem_dim * elem_dim;
			return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ptr, elem_dim, elem_dim);
		}

		span<const int> global_ids(int local_basis_id) const
		{
			const int mapping_id = element_desc.dof_mapping_range.offset + local_basis_id;
			const auto &desc = dof_mapping.mapping_desc[mapping_id];
			return slice_by_range(dof_mapping.node_ids, desc.id_and_weight_range);
		}

		span<const double> global_weights(int local_basis_id) const
		{
			const int mapping_id = element_desc.dof_mapping_range.offset + local_basis_id;
			const auto &desc = dof_mapping.mapping_desc[mapping_id];
			return slice_by_range(dof_mapping.weights, desc.id_and_weight_range);
		}

		template <int dim, typename Derived>
		void gather_unknown(int local_basis_id, Eigen::MatrixBase<Derived> &out) const
		{
			out.derived().setZero();
			auto ids = global_ids(local_basis_id);
			auto weights = global_weights(local_basis_id);
			for (int i = 0; i < ids.size(); ++i)
			{
				for (int d = 0; d < dim; ++d)
				{
					out.derived()(d) += weights[i] * x(ids[i] * dim + d, 0);
				}
			}
		}

		template <int dim, typename Derived>
		void gather_unknowns(Eigen::MatrixBase<Derived> &out) const
		{
			out.derived().setZero(basis_num, dim);
			for (int i = 0; i < basis_num; ++i)
			{
				auto ids = global_ids(i);
				auto weights = global_weights(i);
				for (int ii = 0; ii < ids.size(); ++ii)
				{
					for (int d = 0; d < dim; ++d)
					{
						out.derived()(i, d) += weights[ii] * x(ids[ii] * dim + d, 0);
					}
				}
			}
		}

		template <int dim, typename Derived>
		void gather_previous_unknowns(Eigen::MatrixBase<Derived> &out) const
		{
			out.derived().setZero(basis_num, dim);
			for (int i = 0; i < basis_num; ++i)
			{
				auto ids = global_ids(i);
				auto weights = global_weights(i);
				for (int ii = 0; ii < ids.size(); ++ii)
				{
					for (int d = 0; d < dim; ++d)
					{
						out.derived()(i, d) += weights[ii] * x_prev(ids[ii] * dim + d, 0);
					}
				}
			}
		}

		template <int dim, typename Derived>
		void gather_basis_grads(int quadrature_point_id, Eigen::MatrixBase<Derived> &out) const
		{
			out.derived().resize(basis_num, dim);
			for (int i = 0; i < basis_num; ++i)
			{
				out.derived().row(i) = gather_basis_grad<dim>(i, quadrature_point_id);
			}
		}

		Eigen::VectorXd eval_deformed_jacobian_determinant() const
		{
			// Robust Bezier Jacobian evaluation depends on legacy element bases.
			// The cache-backed NL path does not retain those objects.
			assert(false && "Robust Jacobian is not available in cache-backed NL assembly yet.");
			return Eigen::VectorXd::Ones(quad_num);
		}

	private:
		static const Eigen::MatrixXd &empty_matrix()
		{
			static const Eigen::MatrixXd empty;
			return empty;
		}

		template <int dim>
		Eigen::Matrix<double, 1, dim> gather_row_vector(int offset, span<const double> x, span<const double> y, span<const double> z) const
		{
			Eigen::Matrix<double, 1, dim> result;
			result(0) = x[offset];
			if constexpr (dim >= 2)
				result(1) = y[offset];
			if constexpr (dim >= 3)
				result(2) = z[offset];
			return result;
		}

		RowVectorNd gather_dynamic_row_vector(int offset, span<const double> x, span<const double> y, span<const double> z) const
		{
			RowVectorNd result;
			result.resize(elem_dim);
			result(0) = x[offset];
			if (elem_dim >= 2)
				result(1) = y[offset];
			if (elem_dim >= 3)
				result(2) = z[offset];
			return result;
		}
	};

	class LinearElementAssemblyData
	{
	public:
		int element_id = -1;
		int row_local_basis_id = -1;
		int col_local_basis_id = -1;
		int elem_dim = 0;
		int basis_num = 0;
		int quad_num = 0;
		double time = 0.0;

		span<const double> weighted_measure;
		span<const double> basis_values;

		// Gradient in basis evaluation coordinates:
		// reference/parametric for parameterized bases, physical for non-parametric bases.
		span<const double> basis_grad_x;
		span<const double> basis_grad_y;
		span<const double> basis_grad_z;

		// Always physical-space gradient.
		span<const double> basis_grad_phy_x;
		span<const double> basis_grad_phy_y;
		span<const double> basis_grad_phy_z;

		span<const double> physical_x;
		span<const double> physical_y;
		span<const double> physical_z;

		span<const double> J_inverse_transpose;

		LinearElementAssemblyData() = default;
		LinearElementAssemblyData(
			int element_id,
			int cache_desc_id,
			int row_local_basis_id,
			int col_local_basis_id,
			int dim,
			int basis_num,
			double time,
			const AssemblyCacheView &cache)
			: element_id(element_id),
			  row_local_basis_id(row_local_basis_id),
			  col_local_basis_id(col_local_basis_id),
			  elem_dim(dim),
			  basis_num(basis_num),
			  time(time)
		{
			const AssemblyCacheDesc &desc = cache.desc[cache_desc_id];
			weighted_measure = slice_by_range(cache.weighted_measure, desc.weighted_measure_range);
			basis_values = slice_by_range(cache.basis_values, desc.basis_val_range);
			basis_grad_x = slice_by_range(cache.basis_grad_x, desc.basis_grad_x_range);
			basis_grad_y = slice_by_range(cache.basis_grad_y, desc.basis_grad_y_range);
			basis_grad_z = slice_by_range(cache.basis_grad_z, desc.basis_grad_z_range);
			basis_grad_phy_x = slice_by_range(cache.basis_grad_phy_x, desc.basis_grad_phy_x_range);
			basis_grad_phy_y = slice_by_range(cache.basis_grad_phy_y, desc.basis_grad_phy_y_range);
			basis_grad_phy_z = slice_by_range(cache.basis_grad_phy_z, desc.basis_grad_phy_z_range);
			physical_x = slice_by_range(cache.physical_x, desc.physical_x_range);
			physical_y = slice_by_range(cache.physical_y, desc.physical_y_range);
			physical_z = slice_by_range(cache.physical_z, desc.physical_z_range);
			J_inverse_transpose = slice_by_range(cache.J_inverse_transpose, desc.J_inverse_transpose_range);

			quad_num = weighted_measure.size();
		}

		double gather_basis_value(int local_basis_id, int quadrature_point_id) const
		{
			return basis_values[local_basis_id * quad_num + quadrature_point_id];
		}

		template <int dim>
		Eigen::Matrix<double, dim, 1> gather_basis_grad(int local_basis_id, int quadrature_point_id) const
		{
			int offset = local_basis_id * quad_num + quadrature_point_id;
			return gather_vector<dim>(offset, basis_grad_x, basis_grad_y, basis_grad_z);
		}

		template <int dim>
		Eigen::Matrix<double, dim, 1> gather_basis_grad_phy(int local_basis_id, int quadrature_point_id) const
		{
			int offset = local_basis_id * quad_num + quadrature_point_id;
			return gather_vector<dim>(offset, basis_grad_phy_x, basis_grad_phy_y, basis_grad_phy_z);
		}

		template <int dim>
		Eigen::Matrix<double, dim, 1> gather_phy_position(int quadrature_point_id) const
		{
			int offset = quadrature_point_id;
			return gather_vector<dim>(offset, physical_x, physical_y, physical_z);
		}

		template <int dim>
		void fill_phy_position(int quadrature_point_id, double &x, double &y, double &z) const
		{
			int q = quadrature_point_id;
			if constexpr (dim >= 1)
				x = physical_x[q];
			if constexpr (dim >= 2)
				x = physical_y[q];
			if constexpr (dim >= 3)
				x = physical_z[q];
		};

		template <int dim>
		Eigen::Matrix<double, dim, dim, Eigen::RowMajor> gather_J_inverse_transpose(int quadrature_point_id) const
		{
			const double *ptr = J_inverse_transpose.data() + quadrature_point_id * dim * dim;
			Eigen::Matrix<double, dim, dim, Eigen::RowMajor> mat = Eigen::Map<
				const Eigen::Matrix<double, dim, dim, Eigen::RowMajor>>(ptr);
			return mat;
		}

	private:
		template <int dim>
		Eigen::Matrix<double, dim, 1> gather_vector(int offset, span<const double> x, span<const double> y, span<const double> z) const
		{
			Eigen::Matrix<double, dim, 1> gradient;
			gradient(0) = x[offset];
			if constexpr (dim >= 2)
			{
				gradient(1) = y[offset];
			}
			if constexpr (dim >= 3)
			{
				gradient(2) = z[offset];
			}
			return gradient;
		}
	};

	class MixedAssemblerData
	{
	public:
		MixedAssemblerData(
			const ElementAssemblyValues &psi_vals,
			const ElementAssemblyValues &phi_vals,
			const double t,
			int i, int j,
			const QuadratureVector &da)
			: psi_vals(psi_vals), phi_vals(phi_vals),
			  t(t), i(i), j(j), da(da)
		{
		}

		/// stores the evaluation for that element
		const ElementAssemblyValues &psi_vals;
		/// stores the evaluation for that element
		const ElementAssemblyValues &phi_vals;

		const double t;
		/// first local order
		const int i;
		/// second local order
		const int j;
		/// contains both the quadrature weight and the change of metric in the integral
		const QuadratureVector &da;
	};

	class OptAssemblerData
	{
	public:
		OptAssemblerData(
			const double t,
			const double dt,
			const int el_id,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &global_pts,
			const Eigen::MatrixXd &grad_u_i)
			: t(t), dt(dt), el_id(el_id), local_pts(local_pts), global_pts(global_pts), grad_u_i(grad_u_i)
		{
		}

		const double t;
		const double dt;
		const int el_id;
		const Eigen::MatrixXd &local_pts;
		const Eigen::MatrixXd &global_pts;
		const Eigen::MatrixXd &grad_u_i;
	};

	class OutputData
	{
	public:
		OutputData(
			const double t,
			const int el_id,
			const basis::ElementBases &bs,
			const basis::ElementBases &gbs,
			const Eigen::MatrixXd &local_pts,
			const Eigen::MatrixXd &fun)
			: t(t), el_id(el_id), bs(bs), gbs(gbs), local_pts(local_pts), fun(fun)
		{
		}

		const double t;
		const int el_id;
		const basis::ElementBases &bs;
		const basis::ElementBases &gbs;
		const Eigen::MatrixXd &local_pts;
		const Eigen::MatrixXd &fun;
	};
} // namespace polyfem::assembler
