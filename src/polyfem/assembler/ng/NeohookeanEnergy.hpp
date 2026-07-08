#pragma once

#include <polyfem/materials/LameParameter.hpp>
#include <polyfem/materials/Materials.hpp>
#include <polyfem/utils/AutoDiff.hpp>
#include <polyfem/utils/CudaBoth.hpp>
#include <polyfem/utils/Span.hpp>

#include <Eigen/Core>
#include <sys/stat.h>

namespace polyfem::assembler
{

	template <int dim>
	struct NeoHookeanEnergy
	{
		using Material = polyfem::material::NeoHookean<double>;
		static constexpr int VALUE_DIM = dim;
		static constexpr int DIM = dim;
		static constexpr bool NEED_UNKNOWN_VALUE = false;
		static constexpr bool NEED_UNKNOWN_GRAD = true;

		template <typename Scalar>
		POLYFEM_BOTH static Scalar eval_scalar(
			Span<const Scalar> u,
			Span<const Scalar> gradu,
			const Material &material)
		{
			assert(gradu.size() == dim * dim);

			// Deformation grad type.
			using FType = Eigen::Matrix<Scalar, dim, dim, Eigen::RowMajor>;

			auto [lambda, mu] = material::lambda_mu<dim>(material.lame);
			auto F = Eigen::Map<const FType>(gradu.data()) + FType::Identity();
			Scalar log_J = log(F.determinant());
			// μ/2 [ trace(FF^T)^2 - dim ]  - μ log(J) + λ/2 log(J)^2
			return 0.5 * mu * (F.squaredNorm() - dim) - mu * log_J + 0.5 * lambda * log_J * log_J;
		}
	};

} // namespace polyfem::assembler
