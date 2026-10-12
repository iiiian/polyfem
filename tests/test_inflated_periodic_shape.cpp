#include <polyfem/optimization/OptState.hpp>
#include <polyfem/optimization/AdjointNLProblem.hpp>
#include <polyfem/optimization/BuildFromJson.hpp>
#include <polyfem/optimization/Optimizations.hpp>
#include <polyfem/io/MshWriter.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>

#ifdef POLYFEM_WITH_INFLATOR
#include <inflator/Inflator.hpp>
#endif

using namespace polyfem;

namespace
{
	struct TemporaryDirectory
	{
		std::filesystem::path path = std::filesystem::temp_directory_path()
									 / ("polyfem-inflated-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
		TemporaryDirectory() { std::filesystem::create_directories(path); }
		~TemporaryDirectory() { std::filesystem::remove_all(path); }
	};

	void write_square(const std::filesystem::path &path, const int subdivisions)
	{
		const int width = subdivisions + 1;
		Eigen::MatrixXd vertices(width * width, 2);
		Eigen::MatrixXi cells(2 * subdivisions * subdivisions, 3);
		for (int row = 0; row < width; ++row)
			for (int column = 0; column < width; ++column)
				vertices.row(row * width + column) << double(column) / subdivisions, double(row) / subdivisions;
		for (int row = 0; row < subdivisions; ++row)
			for (int column = 0; column < subdivisions; ++column)
			{
				const int vertex = row * width + column;
				const int cell = 2 * (row * subdivisions + column);
				cells.row(cell) << vertex, vertex + 1, vertex + width + 1;
				cells.row(cell + 1) << vertex, vertex + width + 1, vertex + width;
			}
		io::MshWriter::write(path.string(), vertices, cells, std::vector<int>(cells.rows(), 0), false, true);
	}
}

TEST_CASE("inflated periodic shape schema validation", "[inflator][optimization]")
{
	TemporaryDirectory directory;
	const auto state_path = directory.path / "state.json";
	std::ofstream(state_path) << "{}";
	json args = {
		{"parameters", "auto"},
		{"states", {{{"path", state_path.string()}}}},
		{"variable_to_simulation", {{{"type", "inflated-periodic-shape"}, {"state", {0}}, {"composition", json::array()},
			{"inflation", {{"wire", "wire.obj"},
				{"work_directory", directory.path.string()}, {"initial", {1.0, 0.1}}, {"tiles", 2},
				{"meshing", {{"maxArea", 0.001}, {"marchingSquaresGridSize", 64}, {"forceMSGridSize", true}, {"forceConsistentInterfaceMesh", true}}}}}}}}};
	const json validated = solver::AdjointOptUtils::apply_opt_json_spec(args, true);
	CHECK(validated["variable_to_simulation"][0]["inflation"]["graph_radius"] == 2);
	CHECK(validated["variable_to_simulation"][0]["inflation"]["symmetry"] == "auto");
	CHECK(validated["variable_to_simulation"][0]["inflation"]["meshing"]["cellSize"] == 0.15);

	SECTION("empty state selection")
	{
		args["variable_to_simulation"][0]["state"] = json::array();
	}
	SECTION("nonpositive tile count")
	{
		args["variable_to_simulation"][0]["inflation"]["tiles"] = 0;
	}
	SECTION("nonpositive graph radius")
	{
		args["variable_to_simulation"][0]["inflation"]["graph_radius"] = 0;
	}
	SECTION("missing shape parameters")
	{
		args["variable_to_simulation"][0]["inflation"]["initial"] = {1.0};
	}
	SECTION("nonpositive tetrahedron size")
	{
		args["variable_to_simulation"][0]["inflation"]["meshing"]["cellSize"] = 0;
	}

	CHECK_THROWS_WITH(solver::AdjointOptUtils::apply_opt_json_spec(args, true), Catch::Matchers::ContainsSubstring("Invalid input json"));
}

#ifndef POLYFEM_WITH_INFLATOR
TEST_CASE("inflator configuration requires the optional library", "[inflator]")
{
	const json args = {{"type", "inflated-periodic-shape"}, {"state", {0}}, {"composition", json::array()}};
	CHECK_THROWS_WITH(from_json::build_variable_to_simulation(args, {nullptr}, {nullptr}, {}),
		Catch::Matchers::ContainsSubstring("POLYFEM_WITH_INFLATOR=ON"));
}
#endif

TEST_CASE("differentiable homogenization replaces mesh in place", "[varform][macro_strain][optimization]")
{
	TemporaryDirectory directory;
	write_square(directory.path / "coarse.msh", 2);
	write_square(directory.path / "fine.msh", 3);
	json args = {
		{"geometry", {{{"mesh", (directory.path / "coarse.msh").string()}, {"surface_selection", {{"threshold", 1e-8}}}}}},
		{"space", {{"discr_order", 2}}},
		{"materials", {{"type", "NeoHookean"}, {"E", 1000}, {"nu", 0.3}}},
		{"boundary_conditions", {{"periodic", {{{"boundary_ids", {1, 3}}}, {{"boundary_ids", {2, 4}}}}}}},
		{"constraints", {{"zero_mean", true}, {"macro_displacement_gradient", {{"value", {{0, 0}, {0, -0.02}}}, {"fixed_components", {0, 1, 2, 3}}}}}},
		{"solver", {{"linear", {{"solver", "Eigen::SimplicialLDLT"}}}}},
		{"output", {{"log", {{"level", "off"}, {"quiet", true}}}}}};
	SECTION("zero-mean gauge") {}
	SECTION("Dirichlet gauge")
	{
		args["constraints"]["zero_mean"] = false;
		args["boundary_conditions"]["dirichlet_boundary"] = {{{"id", 1}, {"value", {0, 0}}}};
	}
	auto varform = from_json::build_differentiable_varform(args, 1);
	const auto *identity = varform.get();
	Eigen::MatrixXd solution;
	varform->solve(solution, nullptr, {}, true);
	CHECK(varform->solve_data()->rhs_assembler != nullptr);
	CHECK(varform->solve_data()->al_form.size() == 3);
	const int old_dofs = solution.rows();
	varform->replace_mesh((directory.path / "fine.msh").string());
	CHECK(varform.get() == identity);
	CHECK(varform->solve_data()->nl_problem == nullptr);
	varform->prepare();
	solution.resize(0, 0);
	varform->solve(solution, nullptr, {}, true);
	CHECK(varform->solve_data()->rhs_assembler != nullptr);
	CHECK(varform->solve_data()->al_form.size() == 3);
	CHECK(solution.rows() != old_dofs);
	CHECK(solution.rows() == varform->primary_space().ndof());
	CHECK(solution.allFinite());
	CHECK(varform->displacement_gradient()(1, 1) == Catch::Approx(-0.02));
	varform::InitialConditionOverride initial;
	initial.solution = solution;
	initial.displacement_gradient = varform->displacement_gradient();
	Eigen::MatrixXd continued;
	varform->solve(continued, &initial, {}, true);
	CHECK(varform->solve_data()->rhs_assembler != nullptr);
	CHECK(varform->solve_data()->al_form.size() == 3);
	CHECK((continued - solution).norm() < 1e-7);
}

TEST_CASE("homogenization initial guess dependencies", "[varform][macro_strain][optimization]")
{
	TemporaryDirectory directory;
	write_square(directory.path / "square.msh", 2);
	for (int state = 0; state < 2; ++state)
	{
		const json forward = {
			{"geometry", {{{"mesh", (directory.path / "square.msh").string()}, {"surface_selection", {{"threshold", 1e-8}}}}}},
			{"space", {{"discr_order", 2}}},
			{"materials", {{"type", "NeoHookean"}, {"E", 1000}, {"nu", 0.3}}},
			{"boundary_conditions", {{"periodic", {{{"boundary_ids", {1, 3}}}, {{"boundary_ids", {2, 4}}}}}}},
			{"constraints", {{"zero_mean", true}, {"macro_displacement_gradient", {{"value", {{0, 0}, {0, -0.02 * (2 - state)}}}, {"fixed_components", {3}}}}}},
			{"solver", {{"linear", {{"solver", "Eigen::SimplicialLDLT"}}}}},
			{"output", {{"log", {{"level", "off"}, {"quiet", true}}}}}};
		std::ofstream(directory.path / (std::to_string(state) + ".json")) << forward;
	}
	const json args = {
		{"parameters", "auto"},
		{"states", {{{"path", (directory.path / "0.json").string()}, {"initial_guess", 1}}, {{"path", (directory.path / "1.json").string()}}}},
		{"variable_to_simulation", {{{"type", "periodic-shape"}, {"state", {0, 1}}, {"composition", json::array()}}}},
		{"functionals", {{{"type", "homo_disp_grad"}, {"state", 0}, {"dimensions", {1, 1}}, {"weight", 7}}}},
		{"output", {{"directory", directory.path.string()}, {"log", {{"level", "off"}, {"quiet", true}}}}}};
	OptState opt;
	opt.init(args, false);
	opt.create_varforms(1);
	opt.init_variables();
	SECTION("reverse-index continuation")
	{
		opt.create_problem();
		Eigen::VectorXd parameters;
		opt.initial_guess(parameters);
		CHECK(opt.eval(parameters) == Catch::Approx(-0.28));
		CHECK(opt.diff_caches[1]->disp_grad()(1, 1) == Catch::Approx(-0.02));
		Eigen::VectorXd gradient;
		opt.nl_problem->gradient(parameters, gradient);
		CHECK(gradient.norm() < 1e-8);
	}
	SECTION("out-of-range source")
	{
		opt.args["states"][0]["initial_guess"] = 2;
		CHECK_THROWS_WITH(opt.create_problem(), Catch::Matchers::ContainsSubstring("Invalid initial_guess"));
	}
	SECTION("cyclic sources")
	{
		opt.args["states"][1]["initial_guess"] = 0;
		CHECK_THROWS_WITH(opt.create_problem(), Catch::Matchers::ContainsSubstring("Cyclic initial_guess"));
	}
	SECTION("self dependency")
	{
		opt.args["states"][0]["initial_guess"] = 0;
		CHECK_THROWS_WITH(opt.create_problem(), Catch::Matchers::ContainsSubstring("Cyclic initial_guess"));
	}
	SECTION("invalid tensor indices")
	{
		opt.args["functionals"][0]["dimensions"] = {2, 1};
		CHECK_THROWS_WITH(opt.create_problem(), Catch::Matchers::ContainsSubstring("two valid tensor indices"));
	}
	SECTION("parallel continuation")
	{
		opt.args["solver"]["advanced"]["solve_in_parallel"] = true;
		CHECK_THROWS_WITH(opt.create_problem(), Catch::Matchers::ContainsSubstring("serial static homogenization"));
	}
}

TEST_CASE("inflated periodic shape lifecycle and width gradient", "[.][inflator][optimization]")
{
	const char *configuration = std::getenv("POLYFEM_SHOCK_OPT");
	if (!configuration)
		SKIP("Set POLYFEM_SHOCK_OPT to an optimize.py --prepare-only configuration.");
	std::ifstream input(configuration);
	REQUIRE(input.is_open());
	json args;
	input >> args;
	TemporaryDirectory directory;
	args["root_path"] = configuration;
	args["output"] = {{"directory", directory.path.string()}, {"log", {{"level", "off"}, {"quiet", true}}}};
	args["solver"]["max_threads"] = 1;
	args["variable_to_simulation"][0]["inflation"]["work_directory"] = (directory.path / "meshes").string();
	OptState opt;
	opt.init(args, false);
	opt.create_varforms(1);
	opt.init_variables();
	opt.create_problem();
	const auto identities = opt.varforms;
	const auto caches = opt.diff_caches;
	Eigen::VectorXd parameters;
	opt.initial_guess(parameters);
	const double value = opt.eval(parameters);
	REQUIRE(std::isfinite(value));
	for (int state = 0; state < opt.varforms.size(); ++state)
	{
		CAPTURE(state);
		CHECK(opt.varforms[state]->boundary_state().boundary_nodes.size() == 2);
		CHECK(opt.varforms[state]->solve_data()->rhs_assembler != nullptr);
		Eigen::SimplicialLDLT<StiffnessMatrix> factorization(opt.diff_caches[state]->gradu_h(0));
		REQUIRE(factorization.info() == Eigen::Success);
		CAPTURE(factorization.vectorD().maxCoeff());
		CHECK(factorization.vectorD().minCoeff() > 0);
	}
	Eigen::VectorXd gradient;
	opt.nl_problem->gradient(parameters, gradient);
	REQUIRE(gradient.allFinite());
	const int vertices = opt.varforms[0]->get_mesh().n_vertices();
	Eigen::MatrixXd original_vertices, tangent(vertices, 2);
	opt.varforms[0]->get_vertices(original_vertices);
	Eigen::VectorXd direction = Eigen::VectorXd::Zero(parameters.size());
	const auto lower_bounds = args["solver"]["nonlinear"]["box_constraints"]["bounds"][0].get<std::vector<double>>();
	REQUIRE(lower_bounds.size() == parameters.size());
	const auto first_radius = std::find(lower_bounds.begin(), lower_bounds.end(), 0.02);
	REQUIRE(first_radius != lower_bounds.end());
	const int radius_start = std::distance(lower_bounds.begin(), first_radius);
	const int radius_count = std::count(lower_bounds.begin(), lower_bounds.end(), 0.02);
	REQUIRE(std::all_of(first_radius, first_radius + radius_count, [](const double bound) { return bound == 0.02; }));
	direction.segment(radius_start, radius_count).setOnes();
	for (int vertex = 0; vertex < vertices; ++vertex)
		for (int axis = 0; axis < 2; ++axis)
		{
			Eigen::VectorXd coordinate = Eigen::VectorXd::Zero(2 * vertices);
			coordinate(2 * vertex + axis) = 1;
			tangent(vertex, axis) = direction.dot(opt.variable_to_simulations.data[0]->apply_parametrization_jacobian(coordinate, parameters));
		}
	// Test the imported normal-velocity chain rule on fixed connectivity, excluding discrete remeshing noise.
	const auto evaluate_displaced = [&](const double step) {
		for (auto &varform : opt.varforms)
		{
			varform->set_vertex_positions(original_vertices + step * tangent);
			varform->prepare();
		}
		opt.nl_problem->solve_pde();
		return opt.nl_problem->value(parameters);
	};
	const double shape_epsilon = 1e-5;
	const double shape_plus = evaluate_displaced(shape_epsilon);
	const double shape_minus = evaluate_displaced(-shape_epsilon);
	const double shape_difference = (shape_plus - shape_minus) / (2 * shape_epsilon);
	CAPTURE(gradient.dot(direction), shape_difference);
	CHECK(std::abs(gradient.dot(direction) - shape_difference) < 2e-3 * std::max(1.0, std::abs(shape_difference)));
	evaluate_displaced(0);
	const double epsilon = 1e-4;
	Eigen::VectorXd perturbed = parameters;
	perturbed(0) += epsilon;
	const double plus = opt.eval(perturbed);
	perturbed(0) -= 2 * epsilon;
	const double minus = opt.eval(perturbed);
	const double finite_difference = (plus - minus) / (2 * epsilon);
	CAPTURE(gradient(0), finite_difference);
	CHECK(std::abs(gradient(0) - finite_difference) < 2e-3 * std::max(1.0, std::abs(finite_difference)));
	perturbed = parameters;
	perturbed.segment(radius_start, radius_count).array() += parameters.size() == 23 ? 0.02 : 0.01;
	CHECK(std::isfinite(opt.eval(perturbed)));
	CHECK(opt.varforms[0]->get_mesh().n_vertices() != vertices);
	for (int state = 0; state < opt.varforms.size(); ++state)
	{
		CHECK(opt.varforms[state] == identities[state]);
		CHECK(opt.diff_caches[state] == caches[state]);
		CHECK(opt.diff_caches[state]->size() == 1);
		CHECK(opt.diff_caches[state]->u(0).size() == opt.varforms[state]->primary_space().ndof());
		const double prescribed = opt.varforms[state]->get_args()["constraints"]["macro_displacement_gradient"]["value"][1][1];
		CHECK(opt.diff_caches[state]->disp_grad()(1, 1) == Catch::Approx(prescribed));
	}
	CHECK(std::isfinite(opt.eval(parameters)));
	CHECK(opt.varforms[0]->get_mesh().n_vertices() == vertices);
	opt.nl_problem->gradient(parameters, gradient);
	CHECK(gradient.allFinite());
}

#ifdef POLYFEM_WITH_INFLATOR
TEST_CASE("inflator library homogenization and adjoint in 2D and 3D", "[inflator_library][optimization]")
{
	const int dimension = GENERATE(2, 3);
	CAPTURE(dimension);
	TemporaryDirectory directory;
	const auto wire = directory.path / "wire.obj";
	{
		std::ofstream output(wire);
		output << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv -1 0 0\nv 0 -1 0\n";
		if (dimension == 3)
		{
			output << "v 0 0 1\nv 0 0 -1\n";
		}
		output << "l 1 2\nl 1 3\nl 1 4\nl 1 5\n";
		if (dimension == 3)
		{
			output << "l 1 6\nl 1 7\n";
		}
	}
	inflator::Request request;
	request.type = dimension == 2 ? "2D_orthotropic" : "orthotropic";
	request.wire_path = wire.string();
	request.default_thickness = 0.15;
	const json meshing = {{"maxArea", 0.02}, {"marchingSquaresGridSize", 24}, {"forceMSGridSize", true},
		{"facetSize", 0.2}, {"facetDistance", 0.02}, {"cellSize", 0.4}, {"edgeSize", 0.2}};
	request.meshing_options = meshing.dump();
	const auto mesh = inflator::inflate(request);
	REQUIRE(mesh.dimension == dimension);
	Eigen::MatrixXd vertices(mesh.vertices.size(), dimension);
	Eigen::MatrixXi elements(mesh.elements.size(), dimension + 1);
	for (int vertex = 0; vertex < vertices.rows(); ++vertex)
	{
		for (int axis = 0; axis < dimension; ++axis)
		{
			vertices(vertex, axis) = mesh.vertices[vertex][axis] / 2;
		}
	}
	for (int element = 0; element < elements.rows(); ++element)
	{
		for (int corner = 0; corner <= dimension; ++corner)
		{
			elements(element, corner) = mesh.elements[element][corner];
		}
	}
	const auto mesh_path = directory.path / "initial.msh";
	io::MshWriter::write(mesh_path.string(), vertices, elements, std::vector<int>(elements.rows(), 0), dimension == 3, true);
	json periodic = {{{"boundary_ids", {1, 3}}}, {{"boundary_ids", {2, 4}}}};
	if (dimension == 3)
	{
		periodic.push_back({{"boundary_ids", {5, 6}}});
	}
	std::vector<int> fixed(dimension * dimension);
	std::iota(fixed.begin(), fixed.end(), 0);
	const auto pin_path = directory.path / "pin.txt";
	{
		std::ofstream pin(pin_path);
		pin << 0;
		for (int axis = 0; axis < dimension; ++axis)
		{
			pin << " 0";
		}
		pin << '\n';
	}
	json states = json::array();
	for (int state = 0; state < 2; ++state)
	{
		std::vector<std::vector<double>> macro(dimension, std::vector<double>(dimension, 0));
		macro[1][1] = -0.005 * (state + 1);
		const json forward = {
			{"geometry", {{{"mesh", mesh_path.string()}, {"surface_selection", {{"threshold", 1e-8}}}}}},
			{"space", {{"discr_order", 1}}},
			{"materials", {{"type", "NeoHookean"}, {"E", 1000}, {"nu", 0.3}}},
			{"boundary_conditions", {{"periodic", periodic}, {"dirichlet_boundary", {pin_path.string()}}}},
			{"constraints", {{"macro_displacement_gradient", {{"value", macro}, {"fixed_components", fixed}}}}},
			{"contact", {{"enabled", true}, {"periodic", true}, {"dhat", 0.001}, {"use_convergent_formulation", true}}},
			{"solver", {{"linear", {{"solver", "Eigen::SimplicialLDLT"}}},
				{"nonlinear", {{"grad_norm_tol", 1e-8}, {"norm_type", "Euclidean"}, {"line_search", {{"method", "RobustArmijo"}, {"use_grad_norm_tol", -1}}}}},
				{"augmented_lagrangian", {{"nonlinear", {{"norm_type", "Euclidean"}, {"line_search", {{"method", "RobustArmijo"}, {"use_grad_norm_tol", -1}}}}}}},
				{"contact", {{"barrier_stiffness", 1000}}}}},
			{"output", {{"log", {{"level", "off"}, {"quiet", true}}}}}};
		const auto path = directory.path / (std::to_string(state) + ".json");
		std::ofstream(path) << forward;
		states.push_back({{"path", path.string()}, {"initial_guess", state - 1}});
	}
	std::vector<double> initial = {1};
	initial.insert(initial.end(), mesh.parameters.begin(), mesh.parameters.end());
	const json args = {
		{"parameters", "auto"}, {"states", states},
		{"variable_to_simulation", {{{"type", "inflated-periodic-shape"}, {"state", {0, 1}}, {"composition", json::array()},
			{"inflation", {{"wire", wire.string()}, {"symmetry", dimension == 3 ? "auto" : "orthotropic"}, {"work_directory", (directory.path / "meshes").string()},
				{"initial", initial}, {"tiles", 1}, {"meshing", meshing}}}}}},
		{"functionals", {{{"type", "stress"}, {"state", 1}, {"dimensions", {1, 1}}, {"weight", -1}}}},
		{"solver", {{"max_threads", 1}, {"advanced", {{"enable_slim", false}, {"smooth_line_search", false}}}}},
		{"output", {{"directory", directory.path.string()}, {"log", {{"level", "off"}, {"quiet", true}}}}}};
	OptState opt;
	opt.init(args, false);
	opt.create_varforms(1);
	opt.init_variables();
	opt.create_problem();
	const auto identities = opt.varforms;
	const auto caches = opt.diff_caches;
	Eigen::VectorXd parameters;
	opt.initial_guess(parameters);
	const double value = opt.eval(parameters);
	REQUIRE(std::isfinite(value));
	REQUIRE(value > 0);
	Eigen::VectorXd gradient;
	opt.nl_problem->gradient(parameters, gradient);
	REQUIRE(gradient.allFinite());
	CHECK(opt.diff_caches[0]->disp_grad()(1, 1) == Catch::Approx(-0.005));
	CHECK(opt.diff_caches[1]->disp_grad()(1, 1) == Catch::Approx(-0.01));

	Eigen::MatrixXd original_vertices;
	opt.varforms[0]->get_vertices(original_vertices);
	Eigen::VectorXd direction = Eigen::VectorXd::Zero(parameters.size());
	for (int parameter = 0; parameter < mesh.parameter_types.size(); ++parameter)
	{
		if (mesh.parameter_types[parameter] == inflator::ParameterType::Thickness)
		{
			direction(parameter + 1) = 1;
		}
	}
	Eigen::MatrixXd tangent(original_vertices.rows(), dimension);
	for (int vertex = 0; vertex < tangent.rows(); ++vertex)
	{
		for (int axis = 0; axis < dimension; ++axis)
		{
			Eigen::VectorXd coordinate = Eigen::VectorXd::Zero(tangent.size());
			coordinate(dimension * vertex + axis) = 1;
			tangent(vertex, axis) = direction.dot(opt.variable_to_simulations.data[0]->apply_parametrization_jacobian(coordinate, parameters));
		}
	}
	const auto displaced_value = [&](const double step) {
		for (auto &varform : opt.varforms)
		{
			varform->set_vertex_positions(original_vertices + step * tangent);
			varform->prepare();
		}
		opt.nl_problem->solve_pde();
		return opt.nl_problem->value(parameters);
	};
	const double epsilon = 1e-5;
	const double shape_plus = displaced_value(epsilon);
	const double shape_minus = displaced_value(-epsilon);
	const double shape_difference = (shape_plus - shape_minus) / (2 * epsilon);
	CAPTURE(gradient.dot(direction), shape_difference);
	CHECK(gradient.dot(direction) == Catch::Approx(shape_difference).epsilon(2e-3).margin(1e-6));
	displaced_value(0);
	Eigen::VectorXd perturbed = parameters;
	perturbed(0) += epsilon;
	const double plus = opt.eval(perturbed);
	perturbed(0) -= 2 * epsilon;
	const double minus = opt.eval(perturbed);
	CHECK(gradient(0) == Catch::Approx((plus - minus) / (2 * epsilon)).epsilon(2e-3).margin(1e-6));
	perturbed = parameters + 0.025 * direction;
	CHECK(std::isfinite(opt.eval(perturbed)));
	CHECK(opt.varforms[0]->get_mesh().n_vertices() != original_vertices.rows());
	for (int state = 0; state < 2; ++state)
	{
		CHECK(opt.varforms[state] == identities[state]);
		CHECK(opt.diff_caches[state] == caches[state]);
		CHECK(opt.diff_caches[state]->u(0).rows() == opt.varforms[state]->primary_space().ndof());
	}
	CHECK(opt.eval(parameters) == Catch::Approx(value).epsilon(1e-6));
}
#endif
