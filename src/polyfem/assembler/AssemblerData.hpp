#pragma once

#include <polyfem/assembler/ElementAssemblyValues.hpp>
#include <polyfem/utils/span.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>

#include <Eigen/Core>

#include <cassert>

namespace polyfem::assembler
{
	class NonLinearAssemblerData
	{
	public:
		NonLinearAssemblerData(
			const ElementAssemblyValues &vals,
			const double t,
			const double dt,
			const Eigen::MatrixXd &x,
			const Eigen::MatrixXd &x_prev,
			const QuadratureVector &da)
			: vals(vals), t(t), dt(dt), x(x), x_prev(x_prev), da(da)
		{
		}

		const ElementAssemblyValues &vals;
		const double t;
		const double dt;
		const Eigen::MatrixXd &x;
		const Eigen::MatrixXd &x_prev;
		const QuadratureVector &da;
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
