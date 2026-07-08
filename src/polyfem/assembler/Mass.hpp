#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::assembler
{
	class Mass : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		/// computes and returns local stiffness matrix (1x1) for
		/// bases i,j (where i,j is passed in through data)
		/// ie integral of phi_i * phi_j on the given element
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		/// uses autodiff to compute the rhs for a fabricated solution
		/// in this case it just return pt.getHessian().trace()
		/// pt is the evaluation of the solution at a point
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> compute_rhs(const AutodiffHessianPt &pt) const override;

		/// inialize material parameter
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		/// class that stores and compute density per point
		const Density &density() const { return density_; }

		std::string name() const override { return "Mass"; }
		virtual std::map<std::string, ParamFunc> parameters() const override;
		bool has_ng_assembly_support() const override { return size() >= 1 && size() <= 3; }
		std::optional<BSRSparsityPattern> hessian_sparsity_pattern_ng(
			bool is_volume,
			int n_basis,
			const AssemblyEssentials &bases) const override;
		void assemble_hessian_ng(
			bool is_volume,
			int n_basis,
			const AssemblyEssentials &bases,
			const AssemblyEssentials &geom_bases,
			const AssemblyCache &cache,
			const material::MaterialExprRegistry &materials,
			Span<const double> x,
			Span<const double> x_prev,
			double t,
			double dt,
			BSRMatrix &hessian,
			bool project_to_psd,
			double scale) const override;

	private:
		// class that stores and compute density per point
		Density density_;
	};

	class HRZMass : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		/// computes and returns local stiffness matrix (1x1) for
		/// bases i,j (where i,j is passed in through data)
		/// ie integral of phi_i * phi_j on the given element
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 9, 1>
		assemble(const LinearAssemblerData &data) const override;

		virtual std::map<std::string, ParamFunc> parameters() const override
		{
			std::map<std::string, ParamFunc> res;
			return res;
		}

		std::string name() const override { return "HRZMass"; }
	};
} // namespace polyfem::assembler
