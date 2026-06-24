#pragma once

#include <polyfem/assembler/Assembler.hpp>
#include <polyfem/utils/AutodiffTypes.hpp>

namespace polyfem::assembler
{
	// local assembler for bilaplacian, see laplacian
	//  0 L
	//  L M
	class BilaplacianMain : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		void assemble_element(const LinearElementAssemblyData &, span<double> local_element_matrix) const override
		{
			assert(local_element_matrix.size() == 1);
		}

		std::string name() const override { return "Bilaplacian"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }
	};

	class BilaplacianMixed : public MixedAssembler
	{
	public:
		std::string name() const override { return "BilaplacianMixed"; }

		using MixedAssembler::assemble;

		// res is R
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
		assemble(const MixedAssemblerData &data) const override;

		inline int rows() const override { return 1; }
		inline int cols() const override { return 1; }
	};

	class BilaplacianAux : public LinearAssembler
	{
	public:
		using LinearAssembler::assemble;

		// res is R
		void assemble_element(const LinearElementAssemblyData &data, span<double> local_element_matrix) const override;

		std::string name() const override { return "BilaplacianAux"; }
		std::map<std::string, ParamFunc> parameters() const override { return std::map<std::string, ParamFunc>(); }

	private:
		template <int element_dim>
		void assemble_element_impl(const LinearElementAssemblyData &data, span<double> local_element_matrix) const;
	};
} // namespace polyfem::assembler
