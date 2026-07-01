#include <polyfem/basis/EvalLagrangeBasis.hpp>

#include <polyfem/autogen/auto_p_bases.hpp>
#include <polyfem/autogen/auto_pyramid_bases.hpp>
#include <polyfem/autogen/auto_q_bases_1d_val.hpp>
#include <polyfem/autogen/auto_q_bases_1d_grad.hpp>
#include <polyfem/autogen/auto_q_bases_2d_val.hpp>
#include <polyfem/autogen/auto_q_bases_2d_grad.hpp>
#include <polyfem/autogen/auto_q_bases_3d_val.hpp>
#include <polyfem/autogen/auto_q_bases_3d_grad.hpp>

#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>

#include <cassert>

namespace polyfem::basis
{

	POLYFEM_BOTH int lagrange_basis_count(const BasisDesc &desc)
	{
		assert(desc.basis_family == BasisFamily::Lagrange);
		if (desc.dim == 1)
		{
			return autogen::q_basis_count_1d(desc.order);
		}

		switch (desc.element_kind)
		{
		case ElementKind::Simplex:
		{
			if (desc.dim == 2)
				return autogen::p_basis_count_2d(desc.order);
			if (desc.dim == 3)
				return autogen::p_basis_count_3d(desc.order);
			assert(false);
		}
		case ElementKind::Quad:
		{
			assert(desc.dim == 2);
			return autogen::q_basis_count_2d(desc.order);
		}
		case ElementKind::Hex:
		{
			assert(desc.dim == 3);
			return autogen::q_basis_count_3d(desc.order);
		}
		case ElementKind::Pyramid:
		{
			assert(desc.dim == 3);
			return autogen::pyramid_basis_count_3d(desc.order);
		}
		default:
			assert(false);
		}
	}

	POLYFEM_BOTH void lagrange_basis_values(
		const BasisDesc &desc,
		const BasisStoreView &store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values)
	{
		// Lagrange basis doesn't need additional data from store.
		(void)store;

		int basis_count = lagrange_basis_count(desc);
		int n = x.size(); // quadrature point count
		assert(values.size() == basis_count * x.size());
		assert(desc.dim < 2 || y.size() == x.size());
		assert(desc.dim < 3 || z.size() == x.size());
		assert(desc.basis_family == BasisFamily::Lagrange);

		if (desc.dim == 1)
		{
			for (int i = 0; i < basis_count; ++i)
				autogen::q_basis_value_1d(desc.order, i, x, y, z, values.subspan(i * n, n));
			return;
		}

		switch (desc.element_kind)
		{
		case ElementKind::Simplex:
		{
			if (desc.dim == 2)
			{
				for (int i = 0; i < basis_count; ++i)
					autogen::p_basis_value_2d(desc.is_bernstein, desc.order, i, x, y, z, values.subspan(i * n, n));
			}
			else if (desc.dim == 3)
			{
				for (int i = 0; i < basis_count; ++i)
					autogen::p_basis_value_3d(desc.is_bernstein, desc.order, i, x, y, z, values.subspan(i * n, n));
			}
			else
			{
				assert(false);
			}
			break;
		}
		case ElementKind::Quad:
		{
			assert(desc.dim == 2);
			for (int i = 0; i < basis_count; ++i)
				autogen::q_basis_value_2d(desc.order, i, x, y, z, values.subspan(i * n, n));
			break;
		}
		case ElementKind::Hex:
		{
			assert(desc.dim == 3);
			for (int i = 0; i < basis_count; ++i)
				autogen::q_basis_value_3d(desc.order, i, x, y, z, values.subspan(i * n, n));
			break;
		}
		case ElementKind::Pyramid:
		{
			assert(desc.dim == 3);
			for (int i = 0; i < basis_count; ++i)
				autogen::pyramid_basis_value_3d(desc.order, i, x, y, z, values.subspan(i * n, n));
			break;
		}
		default:
			assert(false);
		}
	}

	POLYFEM_BOTH void lagrange_basis_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		// Lagrange basis doesn't need additional data from store.
		(void)store;

		int basis_count = lagrange_basis_count(desc);
		int n = x.size(); // quadrature count
		assert(grad_x.size() == basis_count * x.size());
		assert(desc.dim < 2 || y.size() == x.size());
		assert(desc.dim < 2 || grad_y.size() == basis_count * x.size());
		assert(desc.dim < 3 || z.size() == x.size());
		assert(desc.dim < 3 || grad_z.size() == basis_count * x.size());
		assert(desc.basis_family == BasisFamily::Lagrange);

		if (desc.dim == 1)
		{
			for (int i = 0; i < basis_count; ++i)
				autogen::q_grad_basis_value_1d(
					desc.order,
					i,
					x,
					y,
					z,
					grad_x.subspan(i * n, n),
					{},
					{});
			return;
		}

		switch (desc.element_kind)
		{
		case ElementKind::Simplex:
		{
			if (desc.dim == 2)
			{
				for (int i = 0; i < basis_count; ++i)
					autogen::p_grad_basis_value_2d(
						desc.is_bernstein,
						desc.order,
						i,
						x,
						y,
						z,
						grad_x.subspan(i * n, n),
						grad_y.subspan(i * n, n),
						{});
			}
			else if (desc.dim == 3)
			{
				for (int i = 0; i < basis_count; ++i)
					autogen::p_grad_basis_value_3d(
						desc.is_bernstein,
						desc.order,
						i,
						x,
						y,
						z,
						grad_x.subspan(i * n, n),
						grad_y.subspan(i * n, n),
						grad_z.subspan(i * n, n));
			}
			else
			{
				assert(false);
			}
			break;
		}
		case ElementKind::Quad:
		{
			assert(desc.dim == 2);
			for (int i = 0; i < basis_count; ++i)
				autogen::q_grad_basis_value_2d(
					desc.order,
					i,
					x,
					y,
					z,
					grad_x.subspan(i * n, n),
					grad_y.subspan(i * n, n),
					{});
			break;
		}
		case ElementKind::Hex:
		{
			assert(desc.dim == 3);
			for (int i = 0; i < basis_count; ++i)
				autogen::q_grad_basis_value_3d(
					desc.order,
					i,
					x,
					y,
					z,
					grad_x.subspan(i * n, n),
					grad_y.subspan(i * n, n),
					grad_z.subspan(i * n, n));
			break;
		}
		case ElementKind::Pyramid:
		{
			assert(desc.dim == 3);
			for (int i = 0; i < basis_count; ++i)
				autogen::pyramid_grad_basis_value_3d(
					desc.order,
					i,
					x,
					y,
					z,
					grad_x.subspan(i * n, n),
					grad_y.subspan(i * n, n),
					grad_z.subspan(i * n, n));
			break;
		}
		default:
			assert(false);
		}
	}

	POLYFEM_BOTH void lagrange_basis_value_and_gradients(
		const BasisDesc &desc,
		BasisStoreView store,
		Span<const double> x,
		Span<const double> y,
		Span<const double> z,
		Span<double> values,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z)
	{
		lagrange_basis_values(desc, store, x, y, z, values);
		lagrange_basis_gradients(desc, store, x, y, z, grad_x, grad_y, grad_z);
	}

} // namespace polyfem::basis
