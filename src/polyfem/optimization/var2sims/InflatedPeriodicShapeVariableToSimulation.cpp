#include "InflatedPeriodicShapeVariableToSimulation.hpp"

#include <polyfem/io/MshWriter.hpp>
#include <polyfem/optimization/AdjointTools.hpp>
#include <polyfem/optimization/VarFormDiff.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <algorithm>
#include <inflator/Inflator.hpp>
#include <filesystem>
#include <fstream>

namespace polyfem::solver
{
	InflatedPeriodicShapeVariableToSimulation::InflatedPeriodicShapeVariableToSimulation(
		std::vector<std::shared_ptr<varform::DifferentiableVarForm>> varforms,
		std::vector<std::shared_ptr<DiffCache>> diff_caches,
		CompositeParametrization parametrization, const json &settings)
		: varforms_(std::move(varforms)),
		  diff_caches_(std::move(diff_caches)),
		  parametrization_(std::move(parametrization)),
		  symmetry_(settings.at("symmetry").get<std::string>()),
		  dimension_(varforms_[0]->get_mesh().dimension()),
		  meshing_(settings.at("meshing")),
		  tiles_(settings.at("tiles").get<int>()),
		  graph_radius_(settings.at("graph_radius").get<int>())
	{
		for (const auto &varform : varforms_)
		{
			if (varform->get_mesh().dimension() != dimension_ || !varform->is_homogenization())
			{
				log_and_throw_adjoint_error("Inflated periodic shape requires homogenization states of the same dimension.");
			}

			const Eigen::MatrixXd offsets = varform->periodic_tile_offsets();
			if (offsets.cols() != dimension_ || Eigen::FullPivLU<Eigen::MatrixXd>(offsets).rank() != dimension_)
			{
				log_and_throw_adjoint_error("Inflated periodic shape requires periodicity in every direction.");
			}
		}

		const auto parameters = settings.at("initial").get<std::vector<double>>();
		initial_ = Eigen::Map<const Eigen::VectorXd>(parameters.data(), parameters.size());

		if (symmetry_ == "auto")
		{
			symmetry_ = dimension_ == 2 ? "doubly_periodic" : "orthotropic";
		}
		wire_ = varforms_[0]->input_path(settings.at("wire"));
		work_directory_ = varforms_[0]->output_file_path(settings.at("work_directory"));

		std::filesystem::create_directories(work_directory_);
	}

	bool InflatedPeriodicShapeVariableToSimulation::affects_varform(const varform::DifferentiableVarForm &target) const
	{
		return std::any_of(varforms_.begin(), varforms_.end(), [&](const auto &varform) { return varform.get() == &target; });
	}

	void InflatedPeriodicShapeVariableToSimulation::update(const Eigen::VectorXd &variables)
	{
		const Eigen::VectorXd parameters = parametrization_.eval(variables);
		if (parameters.size() != initial_.size() || !parameters.allFinite() || parameters(0) <= 0)
		{
			log_and_throw_adjoint_error("Invalid inflated periodic shape parameters.");
		}
		if (parameters.size() == last_parameters_.size() && parameters == last_parameters_)
		{
			return;
		}

		const auto directory = std::filesystem::path(work_directory_) / std::to_string(generation_++);
		if (!std::filesystem::create_directory(directory))
		{
			log_and_throw_adjoint_error("Inflator generation directory already exists: {}", directory.string());
		}

		const std::string options_path = (directory / "meshing.json").string();
		std::ofstream(options_path) << meshing_.dump(4);
		std::ofstream(directory / "parameters.json") << json(std::vector<double>(parameters.data(), parameters.data() + parameters.size())).dump(4);

		inflator::Request request;
		request.type = (dimension_ == 2 ? "2D_" : "") + symmetry_;
		request.wire_path = wire_;
		request.parameters.assign(parameters.data() + 1, parameters.data() + parameters.size());
		request.meshing_options = meshing_.dump();
		request.tiles = tiles_;
		request.graph_radius = graph_radius_;
		const inflator::Mesh mesh = inflator::inflate(request);

		Eigen::MatrixXd vertices(mesh.vertices.size(), dimension_);
		Eigen::MatrixXi cells(mesh.elements.size(), dimension_ + 1);
		for (int vertex = 0; vertex < vertices.rows(); ++vertex)
		{
			for (int axis = 0; axis < dimension_; ++axis)
			{
				vertices(vertex, axis) = mesh.vertices[vertex][axis] * (axis == 0 ? parameters(0) : 1.0) / 2;
			}
		}
		for (int cell = 0; cell < cells.rows(); ++cell)
		{
			for (int corner = 0; corner < cells.cols(); ++corner)
			{
				cells(cell, corner) = mesh.elements[cell][corner];
			}
		}

		shape_jacobian_.setZero(parameters.size(), vertices.size());
		for (int vertex = 0; vertex < vertices.rows(); ++vertex)
		{
			shape_jacobian_(0, dimension_ * vertex) = vertices(vertex, 0) / parameters(0);
		}

		for (int parameter = 1; parameter < parameters.size(); ++parameter)
		{
			const auto &velocity = mesh.shape_velocities[parameter - 1];
			for (int vertex = 0; vertex < vertices.rows(); ++vertex)
			{
				for (int axis = 0; axis < dimension_; ++axis)
				{
					shape_jacobian_(parameter, dimension_ * vertex + axis) = velocity[vertex][axis]
																	* (axis == 0 ? parameters(0) : 1.0) / 2;
				}
			}
		}
		if (!vertices.allFinite() || !shape_jacobian_.allFinite())
		{
			log_and_throw_adjoint_error("Non-finite inflator geometry or shape velocities.");
		}

		periodic_map_ = std::make_unique<PeriodicMeshToMesh>(vertices);
		periodic_parameters_ = periodic_map_->inverse_eval(utils::flatten(vertices));
		periodic_jacobian_.setZero(parameters.size(), periodic_parameters_.size());
		// Periodic coordinates are normalized by the whole tile dimensions; use one representative per periodic vertex.
		std::vector<bool> visited(periodic_map_->n_periodic_dof(), false);
		for (int vertex = 0; vertex < vertices.rows(); ++vertex)
		{
			const int representative = periodic_map_->full_to_periodic(vertex);
			if (visited[representative])
			{
				continue;
			}

			visited[representative] = true;
			for (int axis = 0; axis < dimension_; ++axis)
			{
				periodic_jacobian_.block(1, dimension_ * representative + axis, parameters.size() - 1, 1) =
					shape_jacobian_.block(1, dimension_ * vertex + axis, parameters.size() - 1, 1)
					/ (tiles_ * (axis == 0 ? parameters(0) : 1.0));
			}
		}
		// Width changes only the cell matrix in periodic coordinates: cell_xx = tiles * width.
		periodic_jacobian_(0, periodic_parameters_.size() - dimension_ * dimension_) = tiles_;

		const std::string mesh_path = (directory / "mesh.msh").string();
		io::MshWriter::write(mesh_path, vertices, cells, std::vector<int>(cells.rows(), 0), dimension_ == 3, true);
		for (int state = 0; state < varforms_.size(); ++state)
		{
			varforms_[state]->replace_mesh(mesh_path);
			// Clear mesh-dependent solutions and derivatives in place; objectives retain shared references to this cache.
			*diff_caches_[state] = DiffCache();
		}

		last_parameters_ = parameters;
		adjoint_logger().info("Inflated mesh generation {}: {} vertices, {} elements", generation_ - 1, vertices.rows(), cells.rows());
	}

	void InflatedPeriodicShapeVariableToSimulation::update_state_variables(const Eigen::VectorXd &variables, Eigen::VectorXd &state_variables) const
	{
		log_and_throw_adjoint_error("Generated meshes do not support fixed-connectivity SLIM updates.");
	}

	Eigen::VectorXd InflatedPeriodicShapeVariableToSimulation::compute_adjoint_term(const Eigen::VectorXd &variables) const
	{
		Eigen::VectorXd result = Eigen::VectorXd::Zero(initial_.size());
		for (int state = 0; state < varforms_.size(); ++state)
		{
			Eigen::VectorXd term;
			AdjointTools::dJ_periodic_shape_adjoint_term(*varforms_[state],
														 *diff_caches_[state],
														 *periodic_map_,
														 periodic_parameters_,
														 diff_caches_[state]->u(0),
														 get_adjoint_mat(*varforms_[state], *diff_caches_[state], 0),
														 term);
			result += periodic_jacobian_ * term;
		}

		return parametrization_.apply_jacobian(result, variables);
	}

	int InflatedPeriodicShapeVariableToSimulation::inverse_dof() const
	{
		return parametrization_.inverse_size(initial_.size());
	}

	Eigen::VectorXd InflatedPeriodicShapeVariableToSimulation::inverse_eval() const
	{
		return parametrization_.inverse_eval(initial_);
	}

	Eigen::VectorXd InflatedPeriodicShapeVariableToSimulation::apply_parametrization_jacobian(const Eigen::VectorXd &term, const Eigen::VectorXd &variables) const
	{
		return parametrization_.apply_jacobian(shape_jacobian_ * term, variables);
	}
} // namespace polyfem::solver
