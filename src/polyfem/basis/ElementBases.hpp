#pragma once

#include <polyfem/basis/Basis.hpp>
#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/assembler/AssemblyValues.hpp>
#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>
#include <cstdint>

namespace polyfem::basis::ng
{

	enum class ElementKind : uint8_t
	{
		Simplex,
		Quad,
		Hex,
		Prism,
		Pyramid,
		Polygon,
		Polyhedron
	};

	enum class BasisFamily : uint8_t
	{
		Lagrange, // This represents lagrange + berstein.
		Rational,
		QuadraticSpline,
		PolyRbf,
		Barycentric
	};

	struct BasisDesc
	{
		// Common
		ElementKind element_kind;
		BasisFamily basis_family;
		int order;
		int orderq;
		uint8_t dim;
		int element_basis_num; // new
		// If this id is not -1, central dispatcher is not available.
		// In such case, we will always cache basis/basis grad on host.
		int fallback_eval_callback_id = -1;
		// Callback signature design. Eval basis value and grad for one element:
		// void eval(
		//   Span<const double> point_x, // Quad point x
		//   Span<const double> point_y, // Quad point y. Empty for 1D elem.
		//   Span<const double> point_z, // Quad point z. Empty for 1D/2D elem.
		//   Span<double> values,        // Basis value out. Already the right size.
		//   Span<double> grad_x,        // Basis gradient x out. Already the right size.
		//   Span<double> grad_y,        // Basis gradient y out. Empty for 1D elem.
		//   Span<double> grad_z         // Basis gradient z out. Empty for 1D/2D elem.
		// )
		//
		// For poly basis, both quad point and basis grad lives in physical space.
		// For others, they lives in reference space.

		// Lagrange
		bool is_bernstein;

		// Rational
		Range rational_weight_range;
	};

	struct BasisStoreView
	{
		Span<const double> rational_weights;
		// span of std::function here.
	};

	struct BasisStore
	{
		std::vector<double> rational_weights;
		// vector of std::function here

		BasisStoreView view() const { return BasisStoreView{rational_weights}; }
	};

	struct ElementDesc
	{
		QuadratureDesc quadrature_desc;
		QuadratureDesc mass_quadrature_desc;
		BasisDesc basis_desc;
		Range dof_mapping_range;
		bool has_parameterization;
	};

	struct ElementBasesView
	{
		Span<const ElementDesc> element_desc;
		QuadratureStoreView quadrature;
		QuadratureStoreView mass_quadrature;
		DofMappingStoreView dof_mapping;
		BasisStoreView basis;
	};

	class ElementBases
	{
	public:
		std::vector<ElementDesc> element_desc;
		QuadratureStore quadrature_store;
		QuadratureStore mass_quadrature_store;
		DofMappingStore dof_mapping_store;
		BasisStore basis_store;

		ElementBasesView view() const;

		// ----------------------------------------------
		// Legacy
		// ----------------------------------------------
		using LocalNodeFromPrimitiveFunc = std::function<Eigen::VectorXi(const int local_index, const mesh::Mesh &mesh)>;
		[[deprecated]] std::vector<LocalNodeFromPrimitiveFunc> legacy_local_nodes_form_primitive_callbacks;
	};

	int local_basis_num(const BasisDesc &basis_desc);

	void eval_basis_values_at_points(
		const ElementBasesView &bases,
		int element_id,
		int n_points,
		Span<const double> point_x,
		Span<const double> point_y,
		Span<const double> point_z,
		Span<double> values);

	void eval_basis_gradients_at_points(
		const ElementBasesView &bases,
		int element_id,
		int n_points,
		Span<const double> point_x,
		Span<const double> point_y,
		Span<const double> point_z,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z);

	void eval_basis_values_at_quadrature(
		const ElementBasesView &bases,
		int element_id,
		bool is_mass,
		Span<double> values);

	void eval_basis_gradients_at_quadrature(
		const ElementBasesView &bases,
		int element_id,
		bool is_mass,
		Span<double> grad_x,
		Span<double> grad_y,
		Span<double> grad_z);

	void eval_geometry_mapping(
		const ElementBasesView &geometry_bases,
		int element_id,
		int n_points,
		Span<const double> basis_values,
		Span<const double> basis_grad_x,
		Span<const double> basis_grad_y,
		Span<const double> basis_grad_z,
		Span<double> mapped_x,
		Span<double> mapped_y,
		Span<double> mapped_z,
		Span<double> det_jacobian,
		// Row-major per quadrature point: q * dim * dim + row * dim + col.
		Span<double> jacobian_inverse_transpose);

} // namespace polyfem::basis::ng
