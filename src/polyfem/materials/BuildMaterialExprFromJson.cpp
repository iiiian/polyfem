#include <polyfem/materials/MaterialExprRegistry.hpp>

#include <polyfem/materials/BuildMaterialExprFromJson.hpp>
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/utils/ExpressionValue.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>

#include <nlohmann/json.hpp>

#include <string>
#include <unordered_set>
#include <unordered_map>

namespace polyfem::material
{
	namespace
	{
		// Dispatch from_json factory for a single material type.
		void dispatch_from_json(
			int e, // element id
			const json &material,
			int dim, // mesh dim
			const Units &units,
			const std::string &root_path,
			MaterialExprRegistry &registry)
		{
			const std::string type = material.at("type").get<std::string>();

			// Sum material is handled one level higher. We only allow one level of material composition.
			if (type == "MaterialSum")
			{
				log_and_throw_error("Nested MaterialSum is not supported!");
			}
			else if (type == "LinearElasticity")
				registry.set(e, LinearElasticityExpr::from_json(material, units, root_path));
			else if (type == "HookeLinearElasticity")
			{
				if (dim == 1)
					registry.set(e, HookeLinearElasticityExpr<1>::from_json(material, units, root_path));
				else if (dim == 2)
					registry.set(e, HookeLinearElasticityExpr<2>::from_json(material, units, root_path));
				else if (dim == 3)
					registry.set(e, HookeLinearElasticityExpr<3>::from_json(material, units, root_path));
				else
					log_and_throw_error("Unsupported dimension {} for HookeLinearElasticity", dim);
			}
			else if (type == "SaintVenant")
			{
				if (dim == 1)
					registry.set(e, SaintVenantExpr<1>::from_json(material, units, root_path));
				else if (dim == 2)
					registry.set(e, SaintVenantExpr<2>::from_json(material, units, root_path));
				else if (dim == 3)
					registry.set(e, SaintVenantExpr<3>::from_json(material, units, root_path));
				else
					log_and_throw_error("Unsupported dimension {} for SaintVenant", dim);
			}
			else if (type == "NeoHookean")
				registry.set(e, NeoHookeanExpr::from_json(material, units, root_path));
			else if (type == "IsochoricNeoHookean")
				registry.set(e, IsochoricNeoHookeanExpr::from_json(material, units, root_path));
			else if (type == "IncompressibleLinearElasticity")
				registry.set(e, IncompressibleLinearElasticityExpr::from_json(material, units, root_path));
			else if (type == "FixedCorotational")
				registry.set(e, FixedCorotationalExpr::from_json(material, units, root_path));
			else if (type == "MooneyRivlin")
				registry.set(e, MooneyRivlinExpr::from_json(material, units, root_path));
			else if (type == "MooneyRivlin3Param")
				registry.set(e, MooneyRivlin3ParamExpr::from_json(material, units, root_path));
			else if (type == "MooneyRivlin3ParamSymbolic")
				registry.set(e, MooneyRivlin3ParamSymbolicExpr::from_json(material, units, root_path));
			else if (type == "UnconstrainedOgden")
				registry.set(e, UnconstrainedOgdenExpr::from_json(material, units, root_path));
			else if (type == "IncompressibleOgden")
				registry.set(e, IncompressibleOgdenExpr::from_json(material, units, root_path));
			else if (type == "Stokes")
				registry.set(e, StokesExpr::from_json(material, units, root_path));
			else if (type == "NavierStokes")
				registry.set(e, NavierStokesExpr::from_json(material, units, root_path));
			else if (type == "OperatorSplitting")
				registry.set(e, OperatorSplittingExpr::from_json(material, units, root_path));
			else if (type == "Electrostatics")
				registry.set(e, ElectrostaticsExpr::from_json(material, units, root_path));
			else if (type == "Helmholtz")
				registry.set(e, HelmholtzExpr::from_json(material, units, root_path));
			else if (type == "VolumePenalty")
				registry.set(e, VolumePenaltyExpr::from_json(material, units, root_path));
			else if (type == "HGOFiber")
			{
				if (dim == 1)
					registry.set(e, HGOFiberExpr<1>::from_json(material, units, root_path));
				else if (dim == 2)
					registry.set(e, HGOFiberExpr<2>::from_json(material, units, root_path));
				else if (dim == 3)
					registry.set(e, HGOFiberExpr<3>::from_json(material, units, root_path));
				else
					log_and_throw_error("Unsupported dimension {} for HGOFiber", dim);
			}
			else if (type == "ActiveFiber")
			{
				if (dim == 1)
					registry.set(e, ActiveFiberExpr<1>::from_json(material, units, root_path));
				else if (dim == 2)
					registry.set(e, ActiveFiberExpr<2>::from_json(material, units, root_path));
				else if (dim == 3)
					registry.set(e, ActiveFiberExpr<3>::from_json(material, units, root_path));
				else
					log_and_throw_error("Unsupported dimension {} for ActiveFiber", dim);
			}
			else if (type == "AMIPS")
				registry.set(e, AMIPSExpr::from_json(material, units, root_path));
			else
				log_and_throw_error("Unknown material type '{}'", type);
		}
	} // namespace

	MaterialExprRegistry build_material_expr_registry_from_json(
		const json &materials,
		const mesh::Mesh &mesh,
		const Units &units,
		const std::string &root_path)
	{
		int n_elements = mesh.n_elements();
		assert(n_elements >= 0);
		MaterialExprRegistry registry{n_elements};
		int dim = mesh.dimension();

		// Legacy Material json parsing has two modes: single and multi.
		// In single mode, one material json spec applies to all body. Optional body id
		// field are ignored. To me this seems like a minor bug but whatever.
		// In multi mode, we have an json array of material spec, each corresponds to
		// material of a list of body.

		// Build body_id -> material json map , or detect single-object form.
		std::unordered_map<int, const json *> body_id_to_material_json; // for multi mode.
		const json *single = nullptr;                                   // for single mode.
		if (materials.is_array())
		{
			for (const auto &m : materials)
			{
				for (int id : utils::json_as_array<int>(m.at("id")))
					body_id_to_material_json[id] = &m;
			}
		}
		else
		{
			single = &materials;
		}

		std::unordered_set<int> missing; // body id with no material.
		for (int e = 0; e < n_elements; ++e)
		{
			const json *mj = single; // single mode.
			if (mj == nullptr)       // multi mode.
			{
				int bid = mesh.get_body_id(e);
				auto it = body_id_to_material_json.find(bid);
				if (it == body_id_to_material_json.end())
				{
					missing.insert(bid);
					continue;
				}
				mj = it->second;
			}

			std::string type = mj->at("type").get<std::string>();
			// Sum material is just a json syntactic sugar to set up multiple materials
			// for the same body.
			if (type == "MaterialSum")
			{
				for (const auto &model : mj->value("models", json::array()))
				{
					dispatch_from_json(e, model, dim, units, root_path, registry);
				}
			}
			else
			{
				dispatch_from_json(e, *mj, dim, units, root_path, registry);
			}

			// Density is special. It is an optional field living in each material json.
			if (mj->contains("rho") || mj->contains("density"))
			{
				registry.set(e, DensityExpr::from_json(*mj, units, root_path));
			}
		}

		for (int bid : missing)
		{
			logger().warn("Missing material parameters for body {}", bid);
		}

		return registry;
	}
} // namespace polyfem::material
