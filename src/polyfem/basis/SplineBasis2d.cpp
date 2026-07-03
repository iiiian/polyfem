#include "SplineBasis2d.hpp"

#include "LagrangeBasis2d.hpp"
#include "function/QuadraticBSpline2d.hpp"

#include <polyfem/basis/EvalBasis.hpp>
#include <polyfem/quadrature/QuadQuadrature.hpp>
#include <polyfem/mesh/MeshNodes.hpp>

#include <polyfem/assembler/AssemblerUtils.hpp>

#include <polysolve/linear/Solver.hpp>

#include <polyfem/utils/Types.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/autogen/auto_q_bases.hpp>

#include <Eigen/Sparse>

#include <cassert>
#include <iostream>
#include <vector>
#include <array>
#include <map>

// TODO carefull with simplices

namespace polyfem
{
	using namespace polysolve;
	using namespace Eigen;
	using namespace assembler;
	using namespace mesh;
	using namespace quadrature;

	namespace basis
	{
		namespace
		{
			typedef Matrix<std::vector<int>, 3, 3> SpaceMatrix;
			typedef Matrix<RowVectorNd, 3, 3> NodeMatrix;
			typedef std::vector<std::vector<Local2Global>> LocalDofMappings;

			BasisEvalCallback make_spline_eval_callback(
				const std::array<std::array<double, 4>, 3> h_knots,
				const std::array<std::array<double, 4>, 3> v_knots)
			{
				return [h_knots, v_knots](
						   Span<const double> quad_x,
						   Span<const double> quad_y,
						   Span<const double> quad_z,
						   Span<double> values,
						   Span<double> grad_x,
						   Span<double> grad_y,
						   Span<double> grad_z) {
					assert(quad_y.size() == quad_x.size());
					assert(quad_z.empty());

					const int n_points = quad_x.size();
					Eigen::MatrixXd uv(n_points, 2);
					for (int i = 0; i < n_points; ++i)
					{
						uv(i, 0) = quad_x[i];
						uv(i, 1) = quad_y[i];
					}

					for (int y = 0; y < 3; ++y)
					{
						for (int x = 0; x < 3; ++x)
						{
							const int local_index = y * 3 + x;
							const QuadraticBSpline2d spline(h_knots[x], v_knots[y]);

							Eigen::MatrixXd value;
							spline.interpolate(uv, value);
							for (int q = 0; q < n_points; ++q)
								values[local_index * n_points + q] = value(q);

							Eigen::MatrixXd grad;
							spline.derivative(uv, grad);
							for (int q = 0; q < n_points; ++q)
							{
								grad_x[local_index * n_points + q] = grad(q, 0);
								grad_y[local_index * n_points + q] = grad(q, 1);
							}
						}
					}

					for (int i = 0; i < grad_z.size(); ++i)
						grad_z[i] = 0;
				};
			}

			Eigen::MatrixXd evaluate_basis_values(const assembler::ElementBases &bases, const int element_id, const Eigen::MatrixXd &samples)
			{
				const BasisDesc basis_desc = bases.element_desc[element_id].basis_desc;
				const int n_points = samples.rows();
				const int n_bases = basis_count(basis_desc);
				Eigen::VectorXd values(n_bases * n_points);

				auto sample_x = Span<const double>(samples.col(0).data(), n_points);
				auto sample_y = Span<const double>(samples.col(1).data(), n_points);
				basis_values(
					basis_desc,
					bases.basis.view(),
					sample_x,
					sample_y,
					{},
					Span<double>(values.data(), values.size()));

				Eigen::MatrixXd result(n_points, n_bases);
				for (int i = 0; i < n_bases; ++i)
					result.col(i) = values.segment(i * n_points, n_points);
				return result;
			}

			bool is_complete(const LocalDofMappings &mappings)
			{
				for (const auto &mapping : mappings)
				{
					if (mapping.empty())
						return false;
				}
				return true;
			}

			void append_dof_mappings(const int element_id, const int dim, const LocalDofMappings &mappings, assembler::ElementBases &bases)
			{
				if (mappings.empty())
					return;

				int first_mapping_id = 0;
				for (int j = 0; j < mappings.size(); ++j)
				{
					const auto &mapping = mappings[j];
					assert(!mapping.empty());

					std::vector<int> node_ids;
					std::vector<double> weights;
					std::vector<double> node_positions;
					for (const auto &entry : mapping)
					{
						node_ids.push_back(entry.index);
						weights.push_back(entry.val);
						for (int d = 0; d < dim; ++d)
							node_positions.push_back(entry.node(d));
					}

					const int mapping_id = bases.dof_mapping.append(node_ids, weights, node_positions);
					if (j == 0)
						first_mapping_id = mapping_id;
				}

				bases.element_desc[element_id].dof_mapping_range = Range{first_mapping_id, int(mappings.size())};
			}

			void print_local_space(const SpaceMatrix &space)
			{
				std::stringstream ss;
				ss << std::endl;
				for (int j = 2; j >= 0; --j)
				{
					for (int i = 0; i < 3; ++i)
					{
						if (space(i, j).size() > 0)
						{
							for (std::size_t l = 0; l < space(i, j).size(); ++l)
								ss << space(i, j)[l] << ",";

							ss << "\t";
						}
						else
							ss << "x\t";
					}
					ss << std::endl;
				}

				logger().trace("Local space:\n{}", ss.str());
			}

			int node_id_from_edge_index(const Mesh2D &mesh, MeshNodes &mesh_nodes, const Navigation::Index &index)
			{
				const int face_id = mesh.switch_face(index).face;
				if (face_id >= 0 && mesh.is_cube(face_id))
					return mesh_nodes.node_id_from_face(face_id);

				return mesh_nodes.node_id_from_edge(index.edge);
			}

			void explore_direction(const Navigation::Index &index, const Mesh2D &mesh, MeshNodes &mesh_nodes, const int x, const int y, const bool is_x, const bool invert, LocalBoundary &lb, SpaceMatrix &space, NodeMatrix &node, std::map<int, InterfaceData> &poly_edge_to_data)
			{
				int node_id = node_id_from_edge_index(mesh, mesh_nodes, index);
				// bool real_boundary = mesh.node_id_from_edge_index(index, node_id);

				assert(std::find(space(x, y).begin(), space(x, y).end(), node_id) == space(x, y).end());
				space(x, y).push_back(node_id);
				node(x, y) = mesh_nodes.node_position(node_id);
				assert(node(x, y).size() == 2);

				const int x1 = is_x ? x : (invert ? 2 : 0);
				const int y1 = !is_x ? y : (invert ? 2 : 0);

				const int x2 = is_x ? x : (invert ? 0 : 2);
				const int y2 = !is_x ? y : (invert ? 0 : 2);

				const bool is_boundary = mesh_nodes.is_boundary(node_id);
				const bool is_interface = mesh_nodes.is_interface(node_id);

				if (is_boundary)
				{
					lb.add_boundary_primitive(index.edge, LagrangeBasis2d::quad_edge_local_nodes(2, mesh, index)[1] - 4);
					// lb.add_boundary_primitive(index.edge, LagrangeBasis2d::quadr_quad_edge_local_nodes(mesh, index)[1]-4);
					// bounday_nodes.push_back(node_id);
				}
				else if (is_interface)
				{
					InterfaceData &data = poly_edge_to_data[index.edge];
					data.local_indices.push_back(y * 3 + x);
				}
				else
				{
					assert(!is_boundary && !is_interface);

					Navigation::Index start_index = mesh.switch_face(index);
					assert(start_index.vertex == index.vertex);
					assert(start_index.face >= 0);

					Navigation::Index edge1 = mesh.switch_edge(start_index);
					node_id = node_id_from_edge_index(mesh, mesh_nodes, edge1);
					// if(mesh_nodes.is_boundary(node_id))
					// bounday_nodes.push_back(node_id);

					if (std::find(space(x1, y1).begin(), space(x1, y1).end(), node_id) == space(x1, y1).end())
					{
						space(x1, y1).push_back(node_id);
						node(x1, y1) = mesh_nodes.node_position(node_id);
					}

					Navigation::Index edge2 = mesh.switch_edge(mesh.switch_vertex(start_index));
					node_id = node_id_from_edge_index(mesh, mesh_nodes, edge2);
					// if(mesh_nodes.is_boundary(node_id))
					// bounday_nodes.push_back(node_id);
					if (std::find(space(x2, y2).begin(), space(x2, y2).end(), node_id) == space(x2, y2).end())
					{
						space(x2, y2).push_back(node_id);
						node(x2, y2) = mesh_nodes.node_position(node_id);
						// node(x2, y2).push_back(mesh.node_from_edge_index(edge2));
					}
				}
			}

			void add_id_for_poly(const Navigation::Index &index, const int x1, const int y1, const int x2, const int y2, const SpaceMatrix &space, std::map<int, InterfaceData> &poly_edge_to_data)
			{
				auto it = poly_edge_to_data.find(index.edge);
				if (it != poly_edge_to_data.end())
				{
					InterfaceData &data = it->second;

					assert(space(x1, y1).size() == 1);
					data.local_indices.push_back(y1 * 3 + x1);

					assert(space(x2, y2).size() == 1);
					data.local_indices.push_back(y2 * 3 + x2);
				}
			}

			void build_local_space(const Mesh2D &mesh, MeshNodes &mesh_nodes, const int el_index, SpaceMatrix &space, NodeMatrix &node, std::vector<LocalBoundary> &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data)
			{
				assert(!mesh.is_volume());

				Navigation::Index index;
				// space.setConstant(-1);

				const int el_node_id = mesh_nodes.node_id_from_face(el_index);
				space(1, 1).push_back(el_node_id);
				node(1, 1) = mesh_nodes.node_position(el_node_id);
				// (mesh.node_from_face(el_index));

				LocalBoundary lb(el_index, BoundaryType::QUAD_LINE);

				//////////////////////////////////////////
				index = mesh.get_index_from_face(el_index);
				explore_direction(index, mesh, mesh_nodes, 1, 0, false, false, lb, space, node, poly_edge_to_data);

				//////////////////////////////////////////
				index = mesh.next_around_face(index);
				explore_direction(index, mesh, mesh_nodes, 2, 1, true, false, lb, space, node, poly_edge_to_data);

				//////////////////////////////////////////
				index = mesh.next_around_face(index);
				explore_direction(index, mesh, mesh_nodes, 1, 2, false, true, lb, space, node, poly_edge_to_data);

				//////////////////////////////////////////
				index = mesh.next_around_face(index);
				explore_direction(index, mesh, mesh_nodes, 0, 1, true, true, lb, space, node, poly_edge_to_data);

				//////////////////////////////////////////
				if (mesh_nodes.is_boundary_or_interface(space(1, 2).front()) && mesh_nodes.is_boundary_or_interface(space(2, 1).front()))
				{
					assert(space(2, 2).empty());

					Navigation::Index start_index = mesh.get_index_from_face(el_index);
					start_index = mesh.next_around_face(start_index);
					start_index = mesh.next_around_face(start_index);

					const int node_id = mesh_nodes.node_id_from_vertex(start_index.vertex);
					// mesh.vertex_node_id(start_index.vertex);
					space(2, 2).push_back(node_id);
					node(2, 2) = mesh_nodes.node_position(node_id);
					// node(2,2).push_back(mesh.node_from_vertex(start_index.vertex));

					// bounday_nodes.push_back(node_id);
				}

				if (mesh_nodes.is_boundary_or_interface(space(1, 0).front()) && mesh_nodes.is_boundary_or_interface(space(2, 1).front()))
				{
					assert(space(2, 0).empty());

					Navigation::Index start_index = mesh.get_index_from_face(el_index);
					start_index = mesh.next_around_face(start_index);

					const int node_id = mesh_nodes.node_id_from_vertex(start_index.vertex);
					// mesh.vertex_node_id(start_index.vertex);
					space(2, 0).push_back(node_id);
					node(2, 0) = mesh_nodes.node_position(node_id);
					// .push_back(mesh.node_from_vertex(start_index.vertex));

					// bounday_nodes.push_back(node_id);
				}

				if (mesh_nodes.is_boundary_or_interface(space(1, 2).front()) && mesh_nodes.is_boundary_or_interface(space(0, 1).front()))
				{
					assert(space(0, 2).empty());

					Navigation::Index start_index = mesh.get_index_from_face(el_index);
					start_index = mesh.next_around_face(start_index);
					start_index = mesh.next_around_face(start_index);
					start_index = mesh.next_around_face(start_index);

					// const int node_id = mesh.vertex_node_id(start_index.vertex);
					const int node_id = mesh_nodes.node_id_from_vertex(start_index.vertex);
					space(0, 2).push_back(node_id);
					node(0, 2) = mesh_nodes.node_position(node_id);
					// .push_back(mesh.node_from_vertex(start_index.vertex));

					// bounday_nodes.push_back(node_id);
				}

				if (mesh_nodes.is_boundary_or_interface(space(1, 0).front()) && mesh_nodes.is_boundary_or_interface(space(0, 1).front()))
				{
					Navigation::Index start_index = mesh.get_index_from_face(el_index);

					// const int node_id = mesh.vertex_node_id(start_index.vertex);
					const int node_id = mesh_nodes.node_id_from_vertex(start_index.vertex);
					space(0, 0).push_back(node_id);
					node(0, 0) = mesh_nodes.node_position(node_id);
					//.push_back(mesh.node_from_vertex(start_index.vertex));

					// bounday_nodes.push_back(node_id);
				}

				// print_local_space(space);

				////////////////////////////////////////////////////////////////////////
				index = mesh.get_index_from_face(el_index);
				add_id_for_poly(index, 0, 0, 2, 0, space, poly_edge_to_data);

				index = mesh.next_around_face(index);
				add_id_for_poly(index, 2, 0, 2, 2, space, poly_edge_to_data);

				index = mesh.next_around_face(index);
				add_id_for_poly(index, 2, 2, 0, 2, space, poly_edge_to_data);

				index = mesh.next_around_face(index);
				add_id_for_poly(index, 0, 2, 0, 0, space, poly_edge_to_data);

				if (!lb.empty())
					local_boundary.emplace_back(lb);
			}

			void setup_knots_vectors(MeshNodes &mesh_nodes, const SpaceMatrix &space, std::array<std::array<double, 4>, 3> &h_knots, std::array<std::array<double, 4>, 3> &v_knots)
			{
				// left and right neigh are absent
				if (mesh_nodes.is_boundary_or_interface(space(0, 1).front()) && mesh_nodes.is_boundary_or_interface(space(2, 1).front()))
				{
					h_knots[0] = {{0, 0, 0, 1}};
					h_knots[1] = {{0, 0, 1, 1}};
					h_knots[2] = {{0, 1, 1, 1}};
				}
				// left neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(0, 1).front()))
				{
					h_knots[0] = {{0, 0, 0, 1}};
					h_knots[1] = {{0, 0, 1, 2}};
					h_knots[2] = {{0, 1, 2, 3}};
				}
				// right neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(2, 1).front()))
				{
					h_knots[0] = {{-2, -1, 0, 1}};
					h_knots[1] = {{-1, 0, 1, 1}};
					h_knots[2] = {{0, 1, 1, 1}};
				}
				else
				{
					h_knots[0] = {{-2, -1, 0, 1}};
					h_knots[1] = {{-1, 0, 1, 2}};
					h_knots[2] = {{0, 1, 2, 3}};
				}

				// top and bottom neigh are absent
				if (mesh_nodes.is_boundary_or_interface(space(1, 0).front()) && mesh_nodes.is_boundary_or_interface(space(1, 2).front()))
				{
					v_knots[0] = {{0, 0, 0, 1}};
					v_knots[1] = {{0, 0, 1, 1}};
					v_knots[2] = {{0, 1, 1, 1}};
				}
				// bottom neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(1, 0).front()))
				{
					v_knots[0] = {{0, 0, 0, 1}};
					v_knots[1] = {{0, 0, 1, 2}};
					v_knots[2] = {{0, 1, 2, 3}};
				}
				// top neigh is absent
				else if (mesh_nodes.is_boundary_or_interface(space(1, 2).front()))
				{
					v_knots[0] = {{-2, -1, 0, 1}};
					v_knots[1] = {{-1, 0, 1, 1}};
					v_knots[2] = {{0, 1, 1, 1}};
				}
				else
				{
					v_knots[0] = {{-2, -1, 0, 1}};
					v_knots[1] = {{-1, 0, 1, 2}};
					v_knots[2] = {{0, 1, 2, 3}};
				}
			}

			void basis_for_regular_quad(const SpaceMatrix &space, const NodeMatrix &loc_nodes, LocalDofMappings &b)
			{
				for (int y = 0; y < 3; ++y)
				{
					for (int x = 0; x < 3; ++x)
					{
						if (space(x, y).size() == 1)
						{
							const int global_index = space(x, y).front();
							const Eigen::MatrixXd &node = loc_nodes(x, y);
							assert(node.size() == 2);

							const int local_index = y * 3 + x;
							b[local_index].emplace_back(global_index, node, 1.0);
						}
					}
				}
			}

			void basis_for_irregulard_quad(const int el_id, const Mesh2D &mesh, MeshNodes &mesh_nodes, const SpaceMatrix &space, LocalDofMappings &b)
			{
				for (int y = 0; y < 3; ++y)
				{
					for (int x = 0; x < 3; ++x)
					{
						if (space(x, y).size() > 1)
						{
							const int mpx = 1;
							const int mpy = y;

							const int mmx = x;
							const int mmy = 1;

							std::vector<int> other_indices;
							const auto &center = b[1 * 3 + 1].front();

							const auto &el1 = b[mpy * 3 + mpx].front();
							const auto &el2 = b[mmy * 3 + mmx].front();

							Navigation::Index start_index = mesh.get_index_from_face(el_id);
							bool found = false;
							for (int i = 0; i < 4; ++i)
							{
								other_indices.clear();
								int n_neighs = 0;
								Navigation::Index index = start_index;
								do
								{
									const int f_index = mesh_nodes.node_id_from_face(index.face);
									if (f_index != el1.index && f_index != el2.index && f_index != center.index)
										other_indices.push_back(f_index);

									++n_neighs;
									index = mesh.next_around_vertex(index);
								} while (index.face != start_index.face);
								if (n_neighs != 4)
								{
									found = true;
									break;
								}

								start_index = mesh.next_around_face(start_index);
							}
							assert(found);

							const int local_index = y * 3 + x;
							auto &base = b[local_index];

							const int k = int(other_indices.size()) + 3;

							base.resize(k);

							base[0].index = center.index;
							base[0].val = (4. - k) / k;
							base[0].node = center.node;

							base[1].index = el1.index;
							base[1].val = (4. - k) / k;
							base[1].node = el1.node;

							base[2].index = el2.index;
							base[2].val = (4. - k) / k;
							base[2].node = el2.node;

							for (std::size_t n = 0; n < other_indices.size(); ++n)
							{
								base[3 + n].index = other_indices[n];
								base[3 + n].val = 4. / k;
								base[3 + n].node = mesh_nodes.node_position(other_indices[n]);
							}
						}
					}
				}
			}

			void create_q2_nodes(const Mesh2D &mesh, const int el_index, std::set<int> &vertex_id, std::set<int> &edge_id, LocalDofMappings &b, std::vector<LocalBoundary> &local_boundary, int &n_bases)
			{
				b.resize(9);

				LocalBoundary lb(el_index, BoundaryType::QUAD_LINE);

				Navigation::Index index = mesh.get_index_from_face(el_index);
				for (int j = 0; j < 4; ++j)
				{
					int current_vertex_node_id = -1;
					int current_edge_node_id = -1;
					Eigen::Matrix<double, 1, 2> current_edge_node;
					Eigen::MatrixXd current_vertex_node;

					// auto e2l = LagrangeBasis2d::quadr_quad_edge_local_nodes(mesh, index);
					auto e2l = LagrangeBasis2d::quad_edge_local_nodes(2, mesh, index);

					int vertex_basis_id = e2l[0];
					int edge_basis_id = e2l[1];

					const int opposite_face = mesh.switch_face(index).face;

					// if the edge/vertex is boundary the it is a Q2 edge
					bool is_vertex_q2 = true;
					bool is_vertex_boundary = false;

					Navigation::Index vindex = index;

					do
					{
						if (vindex.face < 0)
						{
							is_vertex_boundary = true;
							break;
						}
						if (mesh.is_spline_compatible(vindex.face))
						{
							is_vertex_q2 = false;
							break;
						}
						vindex = mesh.next_around_vertex(vindex);
					} while (vindex.edge != index.edge);

					if (is_vertex_q2)
					{
						vindex = mesh.switch_face(index);
						do
						{
							if (vindex.face < 0)
							{
								is_vertex_boundary = true;
								break;
							}

							if (mesh.is_spline_compatible(vindex.face))
							{
								is_vertex_q2 = false;
								break;
							}
							vindex = mesh.next_around_vertex(vindex);
						} while (vindex.edge != index.edge);
					}

					const bool is_edge_q2 = opposite_face < 0 || !mesh.is_spline_compatible(opposite_face);

					if (is_edge_q2)
					{
						const bool is_new_edge = edge_id.insert(index.edge).second;

						if (is_new_edge)
						{
							current_edge_node_id = n_bases++;
							current_edge_node = mesh.edge_barycenter(index.edge);

							if (opposite_face < 0)
							{
								// bounday_nodes.push_back(current_edge_node_id);
								lb.add_boundary_primitive(index.edge, edge_basis_id - 4);
							}
						}
					}

					if (is_vertex_q2)
					{
						assert(is_edge_q2);
						const bool is_new_vertex = vertex_id.insert(index.vertex).second;

						if (is_new_vertex)
						{
							current_vertex_node_id = n_bases++;
							current_vertex_node = mesh.point(index.vertex);

							// if(is_vertex_boundary)//mesh.is_vertex_boundary(index.vertex))
							// bounday_nodes.push_back(current_vertex_node_id);
						}
					}

					// init new Q2 nodes
					if (current_vertex_node_id >= 0)
						b[vertex_basis_id].emplace_back(current_vertex_node_id, current_vertex_node, 1.0);

					if (current_edge_node_id >= 0)
						b[edge_basis_id].emplace_back(current_edge_node_id, current_edge_node, 1.0);

					index = mesh.next_around_face(index);
				}

				// central node always present
				const int face_basis_id = 8;
				b[face_basis_id].emplace_back(n_bases++, mesh.face_barycenter(el_index), 1.0);

				if (!lb.empty())
					local_boundary.emplace_back(lb);
			}

			void insert_into_global(const Local2Global &data, std::vector<Local2Global> &vec)
			{
				// ignore small weights
				if (fabs(data.val) < 1e-10)
					return;

				bool found = false;

				for (std::size_t i = 0; i < vec.size(); ++i)
				{
					if (vec[i].index == data.index)
					{
						// logger().trace("{} {} {}", vec[i].val, data.val, fabs(vec[i].val - data.val));
						assert(fabs(vec[i].val - data.val) < 1e-10);
						assert((vec[i].node - data.node).norm() < 1e-10);
						found = true;
						break;
					}
				}

				if (!found)
					vec.push_back(data);
			}

			void assign_q2_weights(const Mesh2D &mesh, const int el_index, const assembler::ElementBases &bases, std::vector<LocalDofMappings> &element_dof_mappings)
			{
				Navigation::Index index = mesh.get_index_from_face(el_index);
				auto &b = element_dof_mappings[el_index];

				for (int j = 0; j < 4; ++j)
				{
					const int opposite_face = mesh.switch_face(index).face;

					if (opposite_face < 0 || !mesh.is_cube(opposite_face))
					{
						index = mesh.next_around_face(index);
						continue;
					}

					// const auto param_p = LagrangeBasis2d::quadr_quad_edge_local_nodes_coordinates(mesh, mesh.switch_face(index));
					// const auto indices = LagrangeBasis2d::quadr_quad_edge_local_nodes(mesh, index);

					const auto indices = LagrangeBasis2d::quad_edge_local_nodes(2, mesh, index);
					Eigen::Matrix<double, 3, 2> param_p;

					{
						Eigen::MatrixXd quad_loc_nodes;
						polyfem::autogen::q_nodes_2d(2, quad_loc_nodes);
						const auto opposite_indices = LagrangeBasis2d::quad_edge_local_nodes(2, mesh, mesh.switch_face(index));
						for (int k = 0; k < 3; ++k)
							param_p.row(k) = quad_loc_nodes.row(opposite_indices[k]);
					}

					const int i0 = indices[0];
					const int i1 = indices[1];
					const int i2 = indices[2];

					const Eigen::MatrixXd eval_p = evaluate_basis_values(bases, opposite_face, param_p);
					const auto &other_mappings = element_dof_mappings[opposite_face];

					for (int i = 0; i < other_mappings.size(); ++i)
					{
						const auto &other_b = other_mappings[i];

						if (other_b.empty())
							continue;

						assert(eval_p.rows() == 3);

						// basis i of element opposite face is zero on this elements
						if (eval_p.col(i).cwiseAbs().maxCoeff() <= 1e-10)
							continue;

						for (std::size_t k = 0; k < other_b.size(); ++k)
						{
							auto glob0 = other_b[k];
							glob0.val *= eval_p(0, i);
							auto glob1 = other_b[k];
							glob1.val *= eval_p(1, i);
							auto glob2 = other_b[k];
							glob2.val *= eval_p(2, i);

							insert_into_global(glob0, b[i0]);
							insert_into_global(glob1, b[i1]);
							insert_into_global(glob2, b[i2]);
						}
					}

					index = mesh.next_around_face(index);
				}
			}

			void setup_data_for_polygons(const Mesh2D &mesh, const int el_index, std::map<int, InterfaceData> &poly_edge_to_data)
			{
				Navigation::Index index = mesh.get_index_from_face(el_index);
				for (int j = 0; j < 4; ++j)
				{
					const int opposite_face = mesh.switch_face(index).face;
					const bool is_neigh_poly = opposite_face >= 0 && mesh.is_polytope(opposite_face);

					if (is_neigh_poly)
					{
						// auto e2l = LagrangeBasis2d::quadr_quad_edge_local_nodes(mesh, index);
						auto e2l = LagrangeBasis2d::quad_edge_local_nodes(2, mesh, index);
						const int vertex_basis_id = e2l[0];
						const int edge_basis_id = e2l[1];
						const int vertex_basis_id2 = e2l[2];

						InterfaceData &data = poly_edge_to_data[index.edge];

						data.local_indices.push_back(edge_basis_id);
						data.local_indices.push_back(vertex_basis_id);
						data.local_indices.push_back(vertex_basis_id2);
					}

					index = mesh.next_around_face(index);
				}
			}
		} // namespace

		int SplineBasis2d::build_bases(const Mesh2D &mesh,
									   const std::string &assembler,
									   const int quadrature_order, const int mass_quadrature_order, assembler::ElementBases &bases, std::vector<LocalBoundary> &local_boundary, std::map<int, InterfaceData> &poly_edge_to_data)
		{
			assert(!mesh.is_volume());

			MeshNodes mesh_nodes(mesh, true, true, 1, 1);

			const int n_els = mesh.n_elements();
			std::vector<LocalDofMappings> element_dof_mappings(n_els);

			local_boundary.clear();

			for (int e = 0; e < n_els; ++e)
			{
				bases.element_desc.push_back(ElementDesc{});
				auto &element_desc = bases.element_desc.back();
				element_desc.has_parameterization = !mesh.is_polytope(e);
				bases.legacy_local_nodes_from_primitive.push_back({});
			}

			for (int e = 0; e < n_els; ++e)
			{
				if (!mesh.is_spline_compatible(e))
					continue;

				SpaceMatrix space;
				NodeMatrix loc_nodes;

				// const int max_local_base =
				build_local_space(mesh, mesh_nodes, e, space, loc_nodes, local_boundary, poly_edge_to_data);
				// n_bases = max(n_bases, max_local_base);

				const int real_order = quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler, 2, AssemblerUtils::BasisType::SPLINE, 2);
				const int real_mass_order = mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", 2, AssemblerUtils::BasisType::SPLINE, 2);

				auto &element_desc = bases.element_desc[e];
				Quadrature quad;
				QuadQuadrature{}.get_quadrature(real_order, quad);
				element_desc.quadrature_desc = bases.quadrature.append(quad);
				QuadQuadrature{}.get_quadrature(real_mass_order, quad);
				element_desc.mass_quadrature_desc = bases.mass_quadrature.append(quad);

				bases.legacy_local_nodes_from_primitive[e] = [e](const int primitive_id, const Mesh &mesh) {
					Eigen::VectorXi res(3);
					const auto &mesh2d = dynamic_cast<const Mesh2D &>(mesh);
					auto index = mesh2d.get_index_from_face(e);
					int le;
					for (le = 0; le < mesh2d.n_face_vertices(e); ++le)
					{
						if (index.edge == primitive_id)
							break;
						index = mesh2d.next_around_face(index);
					}
					assert(index.edge == primitive_id);

					switch (le)
					{
					case 3:
						res << (3 * 0 + 0), (3 * 1 + 0), (3 * 2 + 0);
						break;
					case 0:
						res << (3 * 0 + 0), (3 * 0 + 1), (3 * 0 + 2);
						break;
					case 1:
						res << (3 * 0 + 2), (3 * 1 + 2), (3 * 2 + 2);
						break;
					case 2:
						res << (3 * 2 + 0), (3 * 2 + 1), (3 * 2 + 2);
						break;
					default:
						assert(false);
					}

					return res;
				};

				std::array<std::array<double, 4>, 3> h_knots;
				std::array<std::array<double, 4>, 3> v_knots;

				setup_knots_vectors(mesh_nodes, space, h_knots, v_knots);

				// print_local_space(space);

				auto &basis_desc = element_desc.basis_desc;
				basis_desc.element_kind = ElementKind::Quad;
				basis_desc.basis_family = BasisFamily::Unknown;
				basis_desc.order = 2;
				basis_desc.orderq = 2;
				basis_desc.dim = 2;
				basis_desc.basis_num = 9;
				basis_desc.eval_callback_id = bases.basis.append_eval_callback(make_spline_eval_callback(h_knots, v_knots));
				basis_desc.is_bernstein = false;

				auto &element_mapping = element_dof_mappings[e];
				element_mapping.resize(9);
				basis_for_regular_quad(space, loc_nodes, element_mapping);
				basis_for_irregulard_quad(e, mesh, mesh_nodes, space, element_mapping);
			}

			std::set<int> edge_id;
			std::set<int> vertex_id;

			int n_bases = mesh_nodes.n_nodes();

			for (int e = 0; e < n_els; ++e)
			{
				if (mesh.is_polytope(e) || mesh.is_spline_compatible(e))
					continue;

				const int real_order = quadrature_order > 0 ? quadrature_order : AssemblerUtils::quadrature_order(assembler, 2, AssemblerUtils::BasisType::CUBE_LAGRANGE, 2);
				const int real_mass_order = mass_quadrature_order > 0 ? mass_quadrature_order : AssemblerUtils::quadrature_order("Mass", 2, AssemblerUtils::BasisType::CUBE_LAGRANGE, 2);

				auto &element_desc = bases.element_desc[e];
				Quadrature quad;
				QuadQuadrature{}.get_quadrature(real_order, quad);
				element_desc.quadrature_desc = bases.quadrature.append(quad);
				QuadQuadrature{}.get_quadrature(real_mass_order, quad);
				element_desc.mass_quadrature_desc = bases.mass_quadrature.append(quad);

				auto &basis_desc = element_desc.basis_desc;
				basis_desc.element_kind = ElementKind::Quad;
				basis_desc.basis_family = BasisFamily::Lagrange;
				basis_desc.order = 2;
				basis_desc.orderq = 2;
				basis_desc.dim = 2;
				basis_desc.basis_num = 1; // TODO
				basis_desc.eval_callback_id = -1;
				basis_desc.is_bernstein = false;

				bases.legacy_local_nodes_from_primitive[e] = [e](const int primitive_id, const Mesh &mesh) {
					const auto &mesh2d = dynamic_cast<const Mesh2D &>(mesh);
					auto index = mesh2d.get_index_from_face(e);

					for (int le = 0; le < mesh2d.n_face_vertices(e); ++le)
					{
						if (index.edge == primitive_id)
							break;
						index = mesh2d.next_around_face(index);
					}
					assert(index.edge == primitive_id);

					// const auto indices = LagrangeBasis2d::quadr_quad_edge_local_nodes(mesh2d, index);
					const auto indices = LagrangeBasis2d::quad_edge_local_nodes(2, mesh2d, index);
					Eigen::VectorXi res(indices.size());

					for (size_t i = 0; i < indices.size(); ++i)
						res(i) = indices[i];

					return res;
				};

				create_q2_nodes(mesh, e, vertex_id, edge_id, element_dof_mappings[e], local_boundary, n_bases);
			}

			bool missing_bases = false;
			do
			{
				missing_bases = false;
				for (int e = 0; e < n_els; ++e)
				{
					if (mesh.is_polytope(e) || mesh.is_spline_compatible(e))
						continue;

					auto &b = element_dof_mappings[e];
					if (is_complete(b))
						continue;

					assign_q2_weights(mesh, e, bases, element_dof_mappings);

					missing_bases = missing_bases || is_complete(b);
				}
			} while (missing_bases);

			for (int e = 0; e < n_els; ++e)
			{
				if (mesh.is_polytope(e) || mesh.is_spline_compatible(e))
					continue;
				setup_data_for_polygons(mesh, e, poly_edge_to_data);
			}

			for (int e = 0; e < n_els; ++e)
			{
				if (mesh.is_polytope(e))
					continue;
				append_dof_mappings(e, 2, element_dof_mappings[e], bases);
			}

			return n_bases;
		}

		void SplineBasis2d::fit_nodes(const Mesh2D &mesh, const int n_bases, assembler::ElementBases &gbases)
		{
			(void)mesh;
			(void)n_bases;
			(void)gbases;
			assert(false);
		}
	} // namespace basis
} // namespace polyfem
