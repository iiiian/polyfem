#include <polyfem/State.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/varforms/NonlinearElasticVarForm.hpp>
#include <polyfem/solver/forms/PeriodicContactForm.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <set>

using namespace polyfem;

namespace
{
	class PeriodicContactProbe : public varform::NonlinearElasticStaticVarForm
	{
	public:
		using varform::NonlinearElasticStaticVarForm::prepare;
		const ipc::CollisionMesh &periodic_mesh() const { return periodic_collision_mesh_; }
		const Eigen::VectorXi &periodic_mapping() const { return periodic_collision_mesh_to_basis_; }
		int basis_count() const { return space_.n_bases; }
	};

	std::unique_ptr<mesh::Mesh> tunnel_mesh()
	{
		Eigen::MatrixXd vertices(64, 3);
		std::vector<Eigen::Vector4i> tetrahedra;
		const std::array<std::array<int, 3>, 6> permutations = {{{{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}}, {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}}};
		const std::array<int, 3> strides = {{1, 4, 16}};
		for (int depth = 0; depth < 4; ++depth)
		{
			for (int row = 0; row < 4; ++row)
			{
				for (int column = 0; column < 4; ++column)
				{
					const int vertex = column + 4 * row + 16 * depth;
					vertices.row(vertex) << column / 3.0, row / 3.0, depth / 3.0;
					if (column == 3 || row == 3 || depth == 3 || (column == 1 && row == 1))
					{
						continue;
					}
					for (const auto &permutation : permutations)
					{
						Eigen::Vector4i tetrahedron;
						tetrahedron << vertex, vertex + strides[permutation[0]], vertex + strides[permutation[0]] + strides[permutation[1]], vertex + 21;
						Eigen::Matrix3d edges;
						for (int axis = 0; axis < 3; ++axis)
						{
							edges.col(axis).setZero();
							for (int entry = 0; entry <= axis; ++entry)
							{
								edges(permutation[entry], axis) = 1;
							}
						}
						if (edges.determinant() < 0)
						{
							std::swap(tetrahedron(1), tetrahedron(2));
						}
						tetrahedra.push_back(tetrahedron);
					}
				}
			}
		}
		Eigen::MatrixXi elements(tetrahedra.size(), 4);
		for (int element = 0; element < elements.rows(); ++element)
		{
			elements.row(element) = tetrahedra[element];
		}
		auto mesh = mesh::Mesh::create(vertices, elements);
		mesh->compute_boundary_ids([](const size_t, const std::vector<int> &, const RowVectorNd &point, const bool boundary) {
			if (!boundary)
			{
				return 0;
			}
			for (int axis = 0; axis < 3; ++axis)
			{
				if (std::abs(point(axis)) < 1e-10)
				{
					return 2 * axis + 1;
				}
				if (std::abs(point(axis) - 1) < 1e-10)
				{
					return 2 * axis + 2;
				}
			}
			return 7;
		});
		return mesh;
	}
}

TEST_CASE("3D periodic contact surface and derivatives", "[varform][macro_strain][periodic_contact_3d]")
{
	const int order = GENERATE(1, 2);
	CAPTURE(order);
	const json args = {
		{"geometry", {{"mesh", "in-memory.msh"}}},
		{"space", {{"discr_order", order}}},
		{"materials", {{"type", "NeoHookean"}, {"E", 1000}, {"nu", 0.3}}},
		{"boundary_conditions", {{"periodic", {{{"boundary_ids", {1, 2}}}, {{"boundary_ids", {3, 4}}}, {{"boundary_ids", {5, 6}}}}}}},
		{"constraints", {{"zero_mean", true}, {"macro_displacement_gradient", {{"value", {{0, 0, 0}, {0, -0.01, 0}, {0, 0, 0}}}, {"fixed_components", {0, 1, 2, 3, 4, 5, 6, 7, 8}}}}}},
		{"contact", {{"enabled", true}, {"periodic", true}, {"dhat", 0.01}}},
		{"solver", {{"max_threads", 1}, {"linear", {{"solver", "Eigen::SimplicialLDLT"}}},
			{"nonlinear", {{"norm_type", "Euclidean"}, {"line_search", {{"method", "RobustArmijo"}, {"use_grad_norm_tol", -1}}}}},
			{"augmented_lagrangian", {{"nonlinear", {{"norm_type", "Euclidean"}, {"line_search", {{"method", "RobustArmijo"}, {"use_grad_norm_tol", -1}}}}}}}}},
		{"output", {{"log", {{"level", "off"}, {"quiet", true}}}}}};
	State state;
	state.init(args, true);
	PeriodicContactProbe form;
	form.init("NeoHookean", Units(), state.args, "");
	form.set_mesh(tunnel_mesh());
	form.prepare();
	const auto &surface = form.periodic_mesh();
	REQUIRE(surface.dim() == 3);
	CHECK(surface.faces().rows() == 24 * 8 * order * order);
	CHECK(surface.num_vertices() == 4 * (4 * order) * (6 * order + 1));
	CHECK(form.periodic_mapping().minCoeff() >= 0);
	CHECK(form.periodic_mapping().maxCoeff() < form.basis_count());
	std::set<std::pair<int, int>> edges;
	for (int edge = 0; edge < surface.edges().rows(); ++edge)
	{
		const int first = surface.edges()(edge, 0);
		const int second = surface.edges()(edge, 1);
		CHECK(edges.emplace(std::min(first, second), std::max(first, second)).second);
	}

	solver::PeriodicContactForm contact(surface, form.periodic_mapping(), 0.08, 1,
		false, false, false, false, false, true, ipc::BroadPhaseMethod::HASH_GRID, 1e-6, 1000000);
	contact.set_barrier_stiffness(1);
	Eigen::VectorXd extended = Eigen::VectorXd::Zero(3 * form.basis_count() + 9);
	extended(extended.size() - 9) = -0.9;
	contact.init(extended);
	REQUIRE(contact.value(extended) > 0);
	REQUIRE(!contact.collision_set().empty());
	Eigen::VectorXd gradient;
	contact.first_derivative(extended, gradient);
	StiffnessMatrix hessian;
	contact.set_project_to_psd(false);
	contact.second_derivative(extended, hessian);
	const int coordinate = extended.size() - 9;
	const double epsilon = 1e-6;
	Eigen::VectorXd plus = extended;
	plus(coordinate) += epsilon;
	Eigen::VectorXd minus = extended;
	minus(coordinate) -= epsilon;
	contact.solution_changed(plus);
	const double plus_value = contact.value(plus);
	Eigen::VectorXd plus_gradient;
	contact.first_derivative(plus, plus_gradient);
	contact.solution_changed(minus);
	const double minus_value = contact.value(minus);
	Eigen::VectorXd minus_gradient;
	contact.first_derivative(minus, minus_gradient);
	CHECK(gradient(coordinate) == Catch::Approx((plus_value - minus_value) / (2 * epsilon)).epsilon(1e-5));
	CHECK(((plus_gradient - minus_gradient) / (2 * epsilon) - Eigen::VectorXd(hessian.col(coordinate))).norm() < 1e-4 * hessian.col(coordinate).norm());
	CHECK(contact.is_step_collision_free(extended, plus));

	Eigen::MatrixXd solution;
	form.solve(solution);
	CHECK(solution.rows() == 3 * form.basis_count());
	CHECK(solution.allFinite());
}
