#include <polyfem/State.hpp>
#include <polyfem/legacy/State.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/solver/NLHomoProblem.hpp>
#include <polyfem/solver/forms/lagrangian/MacroStrainLagrangianForm.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/varforms/NonlinearElasticVarForm.hpp>
#include <polyfem/varforms/VarFormFactory.hpp>
#include <polyfem/varforms/diff/DifferentiableVarForm.hpp>

#include "VarFormTestAccess.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <tinyxml2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>

using namespace polyfem;

#if defined(NDEBUG) && !defined(_WIN32)
#define HOMOGENIZATION_TEST_LABEL "[varform][macro_strain]"
#else
#define HOMOGENIZATION_TEST_LABEL "[.][varform][macro_strain]"
#endif

namespace
{
	json affine_macro_args()
	{
		return {
			{"geometry", {{"mesh", "in-memory.msh"}}},
			{"space", {{"discr_order", 1}}},
			{"materials", {{"type", "NeoHookean"}, {"E", 1000}, {"nu", 0.3}}},
			{"boundary_conditions", {{"periodic", {{{"boundary_ids", {1, 2}}}, {{"boundary_ids", {3, 4}}}}}}},
			{"constraints", {{"zero_mean", true}, {"macro_displacement_gradient", {{"value", {{0.01, 0.02}, {-0.01, -0.02}}}, {"fixed_components", {0, 1, 2, 3}}}}}},
			{"solver", {{"max_threads", 1}, {"linear", {{"solver", "Eigen::SimplicialLDLT"}}}, {"nonlinear", {{"grad_norm_tol", 1e-10}}}}},
			{"output", {{"log", {{"level", "error"}, {"quiet", true}}}}}};
	}

	json load_macro_scene(const std::string &path)
	{
		std::ifstream input(path);
		REQUIRE(input.is_open());
		json args;
		input >> args;
		args["root_path"] = path;
		utils::apply_common_params(args);
		args["output"] = {{"log", {{"level", "error"}, {"quiet", true}}}};
		args["solver"]["max_threads"] = 1;
		return args;
	}

	std::unique_ptr<mesh::Mesh> square_mesh()
	{
		Eigen::MatrixXd vertices(9, 2);
		for (int row = 0; row < 3; ++row)
			for (int col = 0; col < 3; ++col)
				vertices.row(row * 3 + col) << col * 0.5, row * 0.5;
		Eigen::MatrixXi triangles(8, 3);
		for (int row = 0; row < 2; ++row)
			for (int col = 0; col < 2; ++col)
			{
				const int vertex = row * 3 + col;
				const int triangle = 2 * (row * 2 + col);
				triangles.row(triangle) << vertex, vertex + 1, vertex + 4;
				triangles.row(triangle + 1) << vertex, vertex + 4, vertex + 3;
			}
		auto mesh = mesh::Mesh::create(vertices, triangles);
		mesh->compute_boundary_ids([](const size_t, const std::vector<int> &, const RowVectorNd &point, const bool boundary) {
			if (!boundary)
				return 0;
			if (std::abs(point.x()) < 1e-12)
				return 1;
			if (std::abs(point.x() - 1) < 1e-12)
				return 2;
			if (std::abs(point.y()) < 1e-12)
				return 3;
			return 4;
		});
		return mesh;
	}

	Eigen::MatrixXd basis_positions(const std::vector<basis::ElementBases> &bases, const int n_bases, const int dim)
	{
		Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(n_bases, dim);
		for (const auto &element : bases)
			for (const auto &basis : element.bases)
				for (const auto &global : basis.global())
					positions.row(global.index) = global.node;
		return positions;
	}

	void check_affine_solution(const varform::VarForm &form, const Eigen::MatrixXd &solution, const Eigen::MatrixXd &gradient)
	{
		const auto debug = test::VarFormTestAccess::debug_data(form);
		REQUIRE(debug.bases != nullptr);
		const Eigen::MatrixXd positions = basis_positions(*debug.bases, debug.n_bases, 2);
		const Eigen::MatrixXd displacement = utils::unflatten(solution, 2);
		const Eigen::MatrixXd differences = displacement.rowwise() - displacement.row(0);
		const Eigen::MatrixXd expected = (positions.rowwise() - positions.row(0)) * gradient.transpose();
		CHECK((differences - expected).cwiseAbs().maxCoeff() < 1e-7);
	}
}

TEST_CASE("macro strain affine patch and reset", "[varform][macro_strain]")
{
	json args = affine_macro_args();
	State state;
	state.init(args, true);
	auto form = std::dynamic_pointer_cast<varform::NonlinearElasticStaticVarForm>(state.variational_formulation);
	REQUIRE(form != nullptr);
	for (const bool zero_load : {false, true})
	{
		CAPTURE(zero_load);
		Eigen::Matrix2d gradient;
		gradient << 0.01, 0.02, -0.01, -0.02;
		if (zero_load)
		{
			gradient.setZero();
			json reset_args = state.args;
			reset_args["constraints"]["macro_displacement_gradient"]["value"] = {{0, 0}, {0, 0}};
			form->init("NeoHookean", Units(), reset_args, "");
		}
		form->set_mesh(square_mesh());
		// CHECK(form->displacement_gradient().rows() == 2);
		// CHECK(form->displacement_gradient().isZero());
		Eigen::MatrixXd solution;
		int callbacks = 0;
		form->solve(solution, nullptr, [&](const int step, const Eigen::MatrixXd &result) {
			++callbacks;
			CHECK(step == 0);
			// CHECK((form->displacement_gradient() - gradient).norm() < 1e-12);
			check_affine_solution(*form, result, gradient);
		});
		CHECK(callbacks == 1);
		// CHECK((form->displacement_gradient() - gradient).norm() < 1e-12);
		check_affine_solution(*form, solution, gradient);
		if (zero_load)
			CHECK(solution.cwiseAbs().maxCoeff() < 1e-7);
	}
}

TEST_CASE("macro strain penalty tolerance and weight updates", "[varform][macro_strain]")
{
	class HomogenizationProbe : public varform::NonlinearElasticStaticVarForm
	{
	public:
		double penalty_weight() const { return solve_data_.strain_al_lagr_form->lagrangian_weight(); }
	};
	json args = affine_macro_args();
	args["constraints"]["macro_displacement_gradient"]["value"] = {{0, 0}, {0, -0.02}};
	args["solver"]["augmented_lagrangian"] = {
		{"initial_weight", 1}, {"max_weight", 1e12}, {"error", 1e-8},
		{"nonlinear", {{"grad_norm_tol", 1e-12}, {"max_iterations", 100}}}};
	State state;
	state.init(args, true);
	HomogenizationProbe form;
	form.init("NeoHookean", Units(), state.args, "");
	form.set_mesh(square_mesh());
	Eigen::MatrixXd solution;
	form.solve(solution);
	CHECK(form.penalty_weight() > 1);
	CHECK(form.penalty_weight() < 1e12);
	Eigen::Matrix2d gradient = Eigen::Matrix2d::Zero();
	gradient(1, 1) = -0.02;
	check_affine_solution(form, solution, gradient);
}

TEST_CASE("macro strain initial penalty follows continued compression", "[varform][macro_strain]")
{
	class HomogenizationProbe : public varform::NonlinearElasticStaticVarForm
	{
	public:
		double penalty_weight() const { return solve_data_.strain_al_lagr_form->lagrangian_weight(); }
		double initial_macro_gradient()
		{
			auto problem = std::dynamic_pointer_cast<solver::NLHomoProblem>(solve_data_.nl_problem);
			Eigen::VectorXd extended = Eigen::VectorXd::Zero(problem->full_size() + 4);
			extended(extended.size() - 1) = -0.1;
			const Eigen::VectorXd reduced = problem->extended_to_reduced(extended);
			problem->solution_changed(reduced);
			Eigen::VectorXd gradient;
			problem->gradient(reduced, gradient);
			return problem->reduced_to_disp_grad(gradient, true)(1, 1);
		}
		using varform::NonlinearElasticStaticVarForm::prepare;
	};
	double max_weight = 10000;
	SECTION("gradient advances compression") {}
	SECTION("penalty stays within configured maximum") { max_weight = 1000; }
	json args = affine_macro_args();
	args["constraints"]["macro_displacement_gradient"]["value"] = {{0, 0}, {0, -0.2}};
	args["solver"]["augmented_lagrangian"] = {
		{"initial_weight", 1}, {"max_weight", max_weight}, {"scaling", 2},
		{"nonlinear", {{"max_iterations", 1}, {"grad_norm_tol", 1e-12}, {"allow_out_of_iterations", false}}}};
	State state;
	state.init(args, true);
	HomogenizationProbe form;
	form.init("NeoHookean", Units(), state.args, "");
	form.set_mesh(square_mesh());
	form.prepare();
	const auto debug = test::VarFormTestAccess::debug_data(form);
	const Eigen::MatrixXd positions = basis_positions(*debug.bases, debug.n_bases, 2);
	varform::InitialConditionOverride initial;
	initial.displacement_gradient = Eigen::Matrix2d::Zero();
	initial.displacement_gradient(1, 1) = -0.1;
	initial.solution = utils::flatten(Eigen::MatrixXd(positions * initial.displacement_gradient.transpose()));
	Eigen::MatrixXd solution;
	// Stop the first AL minimization deliberately to inspect the initial-weight selection before later AL updates.
	CHECK_THROWS_WITH(form.solve(solution, &initial), Catch::Matchers::ContainsSubstring("Reached iteration limit"));
	CHECK(form.penalty_weight() > 1);
	CHECK(form.penalty_weight() <= max_weight);
	if (max_weight == 10000)
		CHECK(form.initial_macro_gradient() >= 0);
	else
	{
		CHECK(form.penalty_weight() == max_weight);
		CHECK(form.initial_macro_gradient() < 0);
	}
}

TEST_CASE("incremental load macro strain affine sequence and reset", "[varform][macro_strain]")
{
	json args = affine_macro_args();
	args["time"] = {{"quasistatic", true}, {"t0", 0.5}, {"dt", 0.25}, {"time_steps", 3}};
	args["constraints"]["macro_displacement_gradient"]["value"] = json::array({{"0.01*t", "0.02*t"}, {"-0.01*t", "-0.02*t"}});
	State state;
	state.init(args, true);
	auto form = std::dynamic_pointer_cast<varform::NonlinearElasticTransientVarForm>(state.variational_formulation);
	REQUIRE(form != nullptr);
	for (const int mode : {0, 1, 2, 0})
	{
		CAPTURE(mode);
		json reset_args = state.args;
		if (mode == 1)
			reset_args["constraints"]["macro_displacement_gradient"]["value"] = {{0.01, 0.02}, {-0.01, -0.02}};
		else if (mode == 2)
			reset_args["constraints"]["macro_displacement_gradient"]["value"] = {{0, 0}, {0, 0}};
		form->init("NeoHookean", Units(), reset_args, "");
		form->set_mesh(square_mesh());
		int callbacks = 0;
		int notifications = 0;
		form->set_time_callback([&](int step, int steps, double time, double end_time) {
			CHECK(step == notifications++);
			CHECK(steps == 3);
			CHECK(time == 0.5 + step * 0.25);
			CHECK(end_time == 1.25);
		});
		Eigen::MatrixXd solution, last_solution;
		form->solve(solution, nullptr, [&](const int step, const Eigen::MatrixXd &result) {
			CHECK(step == callbacks++);
			CHECK(form->embedding_time_integrator() == nullptr);
			Eigen::Matrix2d gradient;
			gradient << 0.01, 0.02, -0.01, -0.02;
			gradient *= mode == 0 ? 0.5 + step * 0.25 : (mode == 1 ? 1.0 : 0.0);
			check_affine_solution(*form, result, gradient);
			last_solution = result;
		});
		CHECK(callbacks == 4);
		CHECK(notifications == 4);
		CHECK((solution - last_solution).norm() == 0);
		if (mode == 2)
			CHECK(solution.cwiseAbs().maxCoeff() < 1e-7);
	}
}

TEST_CASE("incremental load macro strain output", "[varform][macro_strain][output]")
{
	for (const int offset : {0, 5})
	{
		CAPTURE(offset);
		const auto output_dir = std::filesystem::temp_directory_path() / fmt::format("polyfem-incremental-macro-{}", std::chrono::steady_clock::now().time_since_epoch().count());
		json args = affine_macro_args();
		args["time"] = {{"quasistatic", true}, {"t0", 0.5}, {"dt", 0.25}, {"time_steps", 3}};
		args["constraints"]["macro_displacement_gradient"]["value"][1][1] = "-0.02*t";
		args["output"]["directory"] = output_dir.string();
		args["output"]["paraview"] = {{"file_name", "sequence.pvd"}, {"volume", true}};
		args["output"]["advanced"] = {{"save_time_sequence", true}, {"timestep_prefix", "increment_"}};
		args["output"]["data"]["file_index_offset"] = offset;
		State state;
		state.init(args, true);
		state.variational_formulation->set_mesh(square_mesh());
		Eigen::MatrixXd solution;
		state.solve(solution);
		int frames = 0;
		for (const auto &entry : std::filesystem::directory_iterator(output_dir))
			if (entry.path().extension() == ".vtm")
				++frames;
		CHECK(frames == 4);
		for (int step = 0; step <= 3; ++step)
		{
			const auto frame = output_dir / fmt::format("increment_{}.vtm", offset + step);
			tinyxml2::XMLDocument document;
			REQUIRE(document.LoadFile(frame.string().c_str()) == tinyxml2::XML_SUCCESS);
			const auto *root = document.FirstChildElement("VTKFile");
			REQUIRE(root != nullptr);
			const auto *field = root->FirstChildElement("FieldData");
			REQUIRE(field != nullptr);
			const auto *time = field->FirstChildElement("DataArray");
			REQUIRE(time != nullptr);
			CHECK(time->DoubleText() == 0.5 + step * 0.25);
		}
		CHECK(std::filesystem::exists(output_dir / "sequence.pvd"));
		if (offset == 0)
		{
			tinyxml2::XMLDocument document;
			REQUIRE(document.LoadFile((output_dir / "sequence.pvd").string().c_str()) == tinyxml2::XML_SUCCESS);
			const auto *root = document.FirstChildElement("VTKFile");
			REQUIRE(root != nullptr);
			const auto *collection = root->FirstChildElement("Collection");
			REQUIRE(collection != nullptr);
			int step = 0;
			for (const auto *frame = collection->FirstChildElement("DataSet"); frame; frame = frame->NextSiblingElement("DataSet"))
			{
				CHECK(frame->DoubleAttribute("timestep") == 0.5 + step * 0.25);
				CHECK(std::string(frame->Attribute("file")) == fmt::format("increment_{}.vtm", step));
				++step;
			}
			CHECK(step == 4);
		}
		std::filesystem::remove_all(output_dir);
	}
}

TEST_CASE("incremental load macro strain matches legacy periodic contact", HOMOGENIZATION_TEST_LABEL)
{
	json args = load_macro_scene(std::string(POLYFEM_DIFF_DIR) + "/input/homogenize-stress-periodic.json");
	args["time"] = {{"quasistatic", true}, {"dt", 1}, {"time_steps", 2}};
	args["constraints"]["macro_displacement_gradient"]["value"][1][1] = "-0.1*(t+1)";
	args["output"]["advanced"]["save_time_sequence"] = false;
	std::vector<Eigen::MatrixXd> reference;
	legacy::State legacy_state;
	legacy_state.init(args, true);
	legacy_state.load_mesh();
	Eigen::MatrixXd legacy_solution, pressure;
	legacy_state.solve(legacy_solution, pressure, [&](int step, legacy::State &, const Eigen::MatrixXd &result, const Eigen::MatrixXd *, const Eigen::MatrixXd *) {
		CHECK(step == int(reference.size()));
		reference.push_back(utils::unflatten(result, 2));
	});
	REQUIRE(reference.size() == 3);
	const Eigen::MatrixXd legacy_positions = basis_positions(legacy_state.bases, legacy_state.n_bases, 2);

	State state;
	state.init(args, true);
	state.load_mesh();
	auto form = std::dynamic_pointer_cast<varform::NonlinearElasticTransientVarForm>(state.variational_formulation);
	REQUIRE(form != nullptr);
	Eigen::MatrixXd solution;
	int callbacks = 0;
	form->solve(solution, nullptr, [&](const int step, const Eigen::MatrixXd &result) {
		CHECK(step == callbacks++);
		REQUIRE(step < int(reference.size()));
		const auto debug = test::VarFormTestAccess::debug_data(*form);
		const Eigen::MatrixXd positions = basis_positions(*debug.bases, debug.n_bases, 2);
		REQUIRE(positions.rows() == legacy_positions.rows());
		Eigen::MatrixXd expected(positions.rows(), 2);
		for (int node = 0; node < positions.rows(); ++node)
		{
			Eigen::Index match;
			const double distance = (legacy_positions.rowwise() - positions.row(node)).rowwise().squaredNorm().minCoeff(&match);
			REQUIRE(distance < 1e-20);
			expected.row(node) = reference[step].row(match);
		}
		Eigen::MatrixXd actual = utils::unflatten(result, 2);
		actual = (actual.rowwise() - actual.colwise().mean()).eval();
		expected = (expected.rowwise() - expected.colwise().mean()).eval();
		CHECK((actual - expected).norm() / std::max(1.0, expected.norm()) < 1e-6);
	});
	CHECK(callbacks == 3);
}

TEST_CASE("macro strain 3d forward regression", HOMOGENIZATION_TEST_LABEL)
{
	json args = load_macro_scene(std::string(POLYFEM_DATA_DIR) + "/standard/homogenization_3d.json");
	args["solver"]["linear"]["solver"] = "Eigen::SimplicialLDLT";
	REQUIRE(varform::uses_varform_state(args));
	State state;
	state.init(args, true);
	state.load_mesh();
	auto form = std::dynamic_pointer_cast<varform::NonlinearElasticStaticVarForm>(state.variational_formulation);
	REQUIRE(form != nullptr);
	Eigen::MatrixXd solution;
	state.solve(solution);
	const auto stats = form->compute_errors(solution);
	const json actual = {
		{"err_l2", stats.l2_err}, {"err_h1", stats.h1_err}, {"err_h1_semi", stats.h1_semi_err}, {"err_linf", stats.linf_err}, {"err_linf_grad", stats.grad_max_err}, {"err_lp", stats.lp_err}};
	const json &expected = args.at("tests");
	for (const auto &[name, value] : actual.items())
	{
		CAPTURE(name);
		const double reference = expected.at(name);
		CHECK(std::abs(value.get<double>() - reference) / std::max(std::abs(reference), 1e-5) <= expected.value("margin", 1e-5));
	}
	// const Eigen::MatrixXd gradient = form->displacement_gradient();
	// REQUIRE(gradient.rows() == 3);
	// CHECK(std::abs(gradient(1, 1) + 0.15) < 1e-12);
	// CHECK(gradient.allFinite());
}

#if defined(POLYFEM_WITH_OPTIMIZATION) && !defined(_WIN32)
TEST_CASE("macro strain forward and differentiable parity", HOMOGENIZATION_TEST_LABEL)
{
	for (const std::string scene : {"homogenize-stress.json", "homogenize-stress-periodic.json"})
	{
		CAPTURE(scene);
		json args = load_macro_scene(std::string(POLYFEM_DIFF_DIR) + "/input/" + scene);
		Eigen::MatrixXd reference_displacement;
		// Eigen::MatrixXd reference_gradient;
		for (const int mode : {0, 1, 2})
		{
			CAPTURE(mode);
			State state;
			state.init(args, true, mode != 0);
			state.load_mesh();
			auto form = std::dynamic_pointer_cast<varform::NonlinearElasticVarForm>(state.variational_formulation);
			REQUIRE(form != nullptr);
			Eigen::MatrixXd solution;
			int callbacks = 0;
			const auto callback = [&](const int step, const Eigen::MatrixXd &) {
				CHECK(step == 0);
				// CHECK(form->displacement_gradient().rows() == 2);
				++callbacks;
			};
			auto differentiable = std::dynamic_pointer_cast<varform::DifferentiableVarForm>(form);
			CHECK(bool(differentiable) == (mode != 0));
			if (mode == 0)
				form->solve(solution, nullptr, callback);
			else
				differentiable->solve(solution, nullptr, callback, mode == 1);
			CHECK(callbacks == 1);
			Eigen::MatrixXd displacement = utils::unflatten(solution, 2);
			displacement = (displacement.rowwise() - displacement.colwise().mean()).eval();
			// const Eigen::MatrixXd gradient = form->displacement_gradient();
			if (mode == 0)
			{
				reference_displacement = displacement;
				// reference_gradient = gradient;
			}
			else
			{
				CHECK((displacement - reference_displacement).norm() / std::max(1.0, reference_displacement.norm()) < 1e-6);
				// CHECK((gradient - reference_gradient).norm() / std::max(1.0, reference_gradient.norm()) < 1e-6);
			}
		}
	}
}
#endif
