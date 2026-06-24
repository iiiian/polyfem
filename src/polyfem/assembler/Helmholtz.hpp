#pragma once

#include <polyfem/assembler/MatParams.hpp>
#include <polyfem/assembler/Assembler.hpp>

#include <polyfem/utils/AutodiffTypes.hpp>

// local assembler for helmolhz equation, see Laplace
namespace polyfem::assembler
{
	class Helmholtz : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		Helmholtz();

		void assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const override;
		VectorNd compute_rhs(const AutodiffHessianPt &pt) const override;

		Eigen::Matrix<AutodiffScalarGrad, Eigen::Dynamic, 1, 0, 3, 1> kernel(const int dim, const AutodiffGradPt &rvect, const AutodiffScalarGrad &r) const override;

		// sets the k parameter
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		GenericMatParam k() const { return k_; }

		std::string name() const override { return "Helmholtz"; }
		std::map<std::string, ParamFunc> parameters() const override;

	private:
		template <int element_dim>
		void assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const;

		GenericMatParam k_;
	};
} // namespace polyfem::assembler
