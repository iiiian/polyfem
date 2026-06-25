#pragma once

#include <algorithm>
#include <polyfem/basis/Basis.hpp>
#include <polyfem/quadrature/Quadrature.hpp>
#include <polyfem/mesh/Mesh.hpp>

#include <polyfem/assembler/AssemblyValues.hpp>
#include <polyfem/utils/span.hpp>
#include <polyfem/utils/Range.hpp>

#include <vector>
#include <cstdint>

namespace polyfem::basis::ng
{

	struct QuadratureDesc
	{
		int dim = -1;
		Range x_range;
		Range y_range;
		Range z_range;
		Range w_range;
	};

	struct QuadratureStoreView
	{
		span<const double> x;
		span<const double> y;
		span<const double> z;
		span<const double> w;
	};

	struct QuadratureStore
	{
		// NOTES: In future, these data member might resides on device.
		std::vector<double> x; //< Quadrature point coordinate 0.
		std::vector<double> y; //< Quadrature point coordinate 1.
		std::vector<double> z; //< Quadrature point coordinate 2.
		std::vector<double> w; //< Quadrature weight.

		QuadratureStoreView view() const
		{
			return QuadratureStoreView{x, y, z, w};
		}

		/// @brief Append quadrature to the store. Require non-empty quadrature.
		QuadratureDesc append(const quadrature::Quadrature &quad)
		{
			assert(quad.size() != 0);

			// Quadrature class stores:
			// - points: A matrix of size (quad_num x dim). Each row is a quadrature point.
			// - weights: A vector of size quad_num.

			QuadratureDesc desc;
			desc.dim = quad.points.cols();
			// dim == 1, y and z are empty.
			if (desc.dim == 1)
			{
				auto col_x = quad.points.col(0);
				desc.x_range = Range{static_cast<int>(x.size()), static_cast<int>(col_x.size())};
				desc.y_range = {};
				desc.z_range = {};
				desc.w_range = Range{static_cast<int>(w.size()), static_cast<int>(quad.weights.size())};

				x.insert(x.end(), col_x.begin(), col_x.end());
				w.insert(w.end(), quad.weights.begin(), quad.weights.end());
			}
			// dim == 2, z is empty.
			else if (desc.dim == 2)
			{
				auto col_x = quad.points.col(0);
				auto col_y = quad.points.col(1);
				desc.x_range = Range{static_cast<int>(x.size()), static_cast<int>(col_x.size())};
				desc.y_range = Range{static_cast<int>(y.size()), static_cast<int>(col_y.size())};
				desc.z_range = {};
				desc.w_range = Range{static_cast<int>(w.size()), static_cast<int>(quad.weights.size())};

				x.insert(x.end(), col_x.begin(), col_x.end());
				y.insert(y.end(), col_y.begin(), col_y.end());
				w.insert(w.end(), quad.weights.begin(), quad.weights.end());
			}
			else if (desc.dim == 3)
			{
				auto col_x = quad.points.col(0);
				auto col_y = quad.points.col(1);
				auto col_z = quad.points.col(2);
				desc.x_range = Range{static_cast<int>(x.size()), static_cast<int>(col_x.size())};
				desc.y_range = Range{static_cast<int>(y.size()), static_cast<int>(col_y.size())};
				desc.z_range = Range{static_cast<int>(z.size()), static_cast<int>(col_z.size())};
				desc.w_range = Range{static_cast<int>(w.size()), static_cast<int>(quad.weights.size())};

				x.insert(x.end(), col_x.begin(), col_x.end());
				y.insert(y.end(), col_y.begin(), col_y.end());
				z.insert(z.end(), col_z.begin(), col_z.end());
				w.insert(w.end(), quad.weights.begin(), quad.weights.end());
			}
			else
			{
				assert(false && "Invalid dimension");
			}

			return desc;
		}
	};

	struct DofMappingDesc
	{
		Range id_and_weight_range;
		Range node_position_range;
	};

	struct DofMappingStoreView
	{
		span<const DofMappingDesc> mapping_desc;
		span<const int> node_ids;
		span<const double> weights;
		span<const double> node_positions;

		span<const int> get_node_ids(int element_id, int basis_id) const
		{
			auto &desc = mapping_desc[element_id];
			return slice_by_range(node_ids, desc.id_and_weight_range);
		}
		span<const double> get_weights(int element_id, int basis_id) const
		{
			auto &desc = mapping_desc[element_id];
			return slice_by_range(weights, desc.id_and_weight_range);
		}
		span<const double> get_positions(int element_id, int basis_id) const
		{
			auto &desc = mapping_desc[element_id];
			return slice_by_range(node_positions, desc.node_position_range);
		}
	};

	struct DofMappingStore
	{
		std::vector<DofMappingDesc> mapping_desc;
		std::vector<int> node_ids;
		std::vector<double> weights;
		std::vector<double> node_positions;

		DofMappingStoreView view() const
		{
			return DofMappingStoreView{mapping_desc, node_ids, weights, node_positions};
		}

		int append(span<const int> node_ids, span<const double> weights, span<const double> node_positions)
		{
			assert(node_ids.size() != 0 && node_ids.size() == weights.size());
			assert(node_positions.size() != 0);

			DofMappingDesc desc;
			// clang-format off
					desc.id_and_weight_range = Range{static_cast<int>(this->node_ids.size()),
													                 static_cast<int>(node_ids.size())};
					desc.node_position_range = Range{static_cast<int>(this->node_positions.size()),
                                           static_cast<int>(node_positions.size())};
			// clang-format on

			this->node_ids.insert(this->node_ids.end(), node_ids.begin(), node_ids.end());
			this->weights.insert(this->weights.end(), weights.begin(), weights.end());
			this->node_positions.insert(this->node_positions.end(), node_positions.begin(), node_positions.end());
			this->mapping_desc.push_back(desc);
			return this->mapping_desc.size() - 1;
		}
	};

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
		//   span<const double> point_x, // Quad point x
		//   span<const double> point_y, // Quad point y. Empty for 1D elem.
		//   span<const double> point_z, // Quad point z. Empty for 1D/2D elem.
		//   span<double> values,        // Basis value out. Already the right size.
		//   span<double> grad_x,        // Basis gradient x out. Already the right size.
		//   span<double> grad_y,        // Basis gradient y out. Empty for 1D elem.
		//   span<double> grad_z         // Basis gradient z out. Empty for 1D/2D elem.
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
		span<const double> rational_weights;
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
		span<const ElementDesc> element_desc;
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
		span<const double> point_x,
		span<const double> point_y,
		span<const double> point_z,
		span<double> values);

	void eval_basis_gradients_at_points(
		const ElementBasesView &bases,
		int element_id,
		int n_points,
		span<const double> point_x,
		span<const double> point_y,
		span<const double> point_z,
		span<double> grad_x,
		span<double> grad_y,
		span<double> grad_z);

	void eval_basis_values_at_quadrature(
		const ElementBasesView &bases,
		int element_id,
		bool is_mass,
		span<double> values);

	void eval_basis_gradients_at_quadrature(
		const ElementBasesView &bases,
		int element_id,
		bool is_mass,
		span<double> grad_x,
		span<double> grad_y,
		span<double> grad_z);

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
		// Row-major per quadrature point: q * dim * dim + row * dim + col.
		span<double> jacobian_inverse_transpose);

} // namespace polyfem::basis::ng
