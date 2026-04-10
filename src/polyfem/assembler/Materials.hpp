#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <vector>

namespace polyfem::assembler
{
	using MaterialExpressionList = std::vector<utils::ExpressionValue>;
	using MaterialExpressionMatrix = std::vector<MaterialExpressionList>;

	struct NeoHookeanMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue E;
		utils::ExpressionValue nu;
		utils::ExpressionValue lambda;
		utils::ExpressionValue mu;
		utils::ExpressionValue rho;
		utils::ExpressionValue phi;
		utils::ExpressionValue psi;
	};

	struct IsochoricNeoHookeanMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue E;
		utils::ExpressionValue nu;
		utils::ExpressionValue lambda;
		utils::ExpressionValue mu;
		utils::ExpressionValue rho;
		utils::ExpressionValue phi;
		utils::ExpressionValue psi;
	};

	struct MooneyRivlinMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue c1;
		utils::ExpressionValue c2;
		utils::ExpressionValue k;
		utils::ExpressionValue rho;
	};

	struct MooneyRivlin3ParamMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue c1;
		utils::ExpressionValue c2;
		utils::ExpressionValue c3;
		utils::ExpressionValue d1;
		utils::ExpressionValue rho;
	};

	struct MooneyRivlin3ParamSymbolicMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue c1;
		utils::ExpressionValue c2;
		utils::ExpressionValue c3;
		utils::ExpressionValue d1;
		utils::ExpressionValue rho;
	};

	struct UnconstrainedOgdenMaterial
	{
		std::vector<int> ids = {0};
		MaterialExpressionList alphas;
		MaterialExpressionList mus;
		MaterialExpressionList Ds;
		utils::ExpressionValue rho;
	};

	struct IncompressibleOgdenMaterial
	{
		std::vector<int> ids = {0};
		MaterialExpressionList c;
		MaterialExpressionList m;
		utils::ExpressionValue k;
		utils::ExpressionValue rho;
	};

	struct LinearElasticityMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue E;
		utils::ExpressionValue nu;
		utils::ExpressionValue lambda;
		utils::ExpressionValue mu;
		utils::ExpressionValue rho;
		utils::ExpressionValue phi;
		utils::ExpressionValue psi;
	};

	struct HookeLinearElasticityMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue E;
		utils::ExpressionValue nu;
		MaterialExpressionList elasticity_tensor;
		utils::ExpressionValue rho;
		MaterialExpressionMatrix fiber_direction;
	};

	struct SaintVenantMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue E;
		utils::ExpressionValue nu;
		MaterialExpressionList elasticity_tensor;
		utils::ExpressionValue rho;
		utils::ExpressionValue phi;
		MaterialExpressionMatrix fiber_direction;
		utils::ExpressionValue psi;
	};

	struct StokesMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue viscosity;
		utils::ExpressionValue rho;
	};

	struct NavierStokesMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue viscosity;
		utils::ExpressionValue rho;
	};

	struct OperatorSplittingMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue viscosity;
		utils::ExpressionValue rho;
	};

	struct ElectrostaticsMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue epsilon;
		utils::ExpressionValue rho;
	};

	struct IncompressibleLinearElasticityMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue E;
		utils::ExpressionValue nu;
		utils::ExpressionValue lambda;
		utils::ExpressionValue mu;
		utils::ExpressionValue rho;
	};

	struct MaterialSumMaterial
	{
		std::vector<int> ids = {0};
		std::vector<json> models;
		utils::ExpressionValue rho;
	};

	struct LaplacianMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue rho;
	};

	struct HelmholtzMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue k;
		utils::ExpressionValue rho;
	};

	struct BilaplacianMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue rho;
	};

	struct AMIPSMaterial
	{
		std::vector<int> ids = {0};
		bool use_rest_pose = false;
		utils::ExpressionValue rho;
	};

	struct FixedCorotationalMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue E;
		utils::ExpressionValue nu;
		utils::ExpressionValue lambda;
		utils::ExpressionValue mu;
		utils::ExpressionValue rho;
		utils::ExpressionValue phi;
		utils::ExpressionValue psi;
	};

	struct VolumePenaltyMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue k;
		utils::ExpressionValue rho;
	};

	struct HGOFiberMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue k1;
		utils::ExpressionValue k2;
		utils::ExpressionValue rho;
		MaterialExpressionMatrix fiber_direction;
	};

	struct ActiveFiberMaterial
	{
		std::vector<int> ids = {0};
		utils::ExpressionValue activation;
		utils::ExpressionValue rho;
		utils::ExpressionValue Tmax;
		MaterialExpressionMatrix fiber_direction;
	};

} // namespace polyfem::assembler
