#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/materials/LameParameter.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <Eigen/Core>

#include <string>

namespace polyfem::material
{
	template <typename Scalar, int dim>
	struct ElasticityTensor
	{
		static_assert(dim >= 1 && dim <= 3, "ElasticityTensor dimension must be 1, 2, or 3");

		/// Voigt order is [xx] in 1D, [xx, yy, xy] in 2D, and [xx, yy, zz, yz, zx, xy] in 3D.
		static constexpr int VOIGT_SIZE = dim == 3 ? 6 : dim == 2 ? 3
																  : 1;

		/// True if elastic tensor should be compute from elastic lame parameters.
		bool depends_on_elastic_coeff = true;

		/// Symmetric stiffness matrix in Voigt notation.
		Eigen::Matrix<Scalar, VOIGT_SIZE, VOIGT_SIZE, Eigen::RowMajor> stiffness;

		/// Builds an isotropic stiffness tensor from Lame parameters.
		static ElasticityTensor<double, dim> from_lame(const LameParameter<double> &lame);

		/// Parses elasticity_tensor JSON arrays.
		/// Accepted layouts:
		///   dim=1: [C00]
		///   dim=2: [Ex, Ey, nuXY, muXY] or [C00, C01, C02, C11, C12, C22]
		///   dim=3: [Et, Ea, nu_t, nu_a, Ga], [Ex, Ey, Ez, nuXY, nuXZ, nuYZ, muYZ, muZX, muXY],
		///          or upper-triangular stiffness entries
		///          [C00, C01, C02, C03, C04, C05, C11, C12, C13, C14, C15,
		///           C22, C23, C24, C25, C33, C34, C35, C44, C45, C55].
		static ElasticityTensor<utils::ExpressionValue, dim> from_json(const json &value, const Units &units, const std::string &root_path);

		/// Evaluate elastic tensor expression at position (x,y,z), time t, and element id.
		/// If depends_on_elastic_coeff, return default ElasticityTensor<double>.
		ElasticityTensor<double, dim> eval_expr(double x, double y, double z = 0, double t = 0, int element_id = -1) const;
	};
} // namespace polyfem::material
