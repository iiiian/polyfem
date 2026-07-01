#include <polyfem/materials/ElasticityTensor.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/Units.hpp>
#include <polyfem/materials/LameParameter.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

#include <Eigen/Core>
#include <Eigen/LU>
#include <spdlog/fmt/fmt.h>

#include <string>
#include <stdexcept>
#include <vector>

namespace polyfem::material
{
	namespace
	{
		std::vector<double> numeric_entries(const json &value, const std::string &field)
		{
			std::vector<double> entries;
			entries.reserve(value.size());
			for (std::size_t i = 0; i < value.size(); ++i)
			{
				if (!value[i].is_number())
				{
					throw std::runtime_error(fmt::format(
						"{}: shortcut form requires numeric entries; use direct stiffness entries for expression-valued tensors",
						field));
				}
				entries.push_back(value[i].get<double>());
			}
			return entries;
		}

		template <int N>
		void build_expr_matrix_from_double(
			const Eigen::Matrix<double, N, N> &in,
			const std::string &unit_type,
			Eigen::Matrix<utils::ExpressionValue, N, N, Eigen::RowMajor> &out)
		{
			for (int i = 0; i < N; ++i)
			{
				for (int j = 0; j < N; ++j)
				{
					out(i, j).init(in(i, j));
					out(i, j).set_unit_type(unit_type);
				}
			}
		}

		template <int N>
		void parse_upper_symmetric_entries(
			const json &value,
			const std::string &root_path,
			const std::string &field,
			const std::string &unit_type,
			Eigen::Matrix<utils::ExpressionValue, N, N, Eigen::RowMajor> &out)
		{
			constexpr std::size_t expected_size = N * (N + 1) / 2;
			if (value.size() != expected_size)
			{
				throw std::runtime_error(fmt::format(
					"{}: expected exactly {} upper-triangular entries, got {}",
					field,
					expected_size,
					value.size()));
			}

			std::size_t entry = 0;
			for (int i = 0; i < N; ++i)
			{
				for (int j = i; j < N; ++j)
				{
					out(i, j).init(value[entry], root_path);
					out(i, j).set_unit_type(unit_type);
					if (i != j)
					{
						out(j, i).init(value[entry], root_path);
						out(j, i).set_unit_type(unit_type);
					}
					++entry;
				}
			}
		}

		Eigen::Matrix<double, 3, 3> orthotropic_stiffness_2d(const std::vector<double> &entries)
		{
			double Ex = entries[0];
			double Ey = entries[1];
			double nuXY = entries[2];
			double muXY = entries[3];
			double nuYX = nuXY * Ey / Ex;

			Eigen::Matrix<double, 3, 3> compliance;
			compliance << 1.0 / Ex, -nuYX / Ey, 0.0,
				-nuXY / Ex, 1.0 / Ey, 0.0,
				0.0, 0.0, 1.0 / (2.0 * muXY);
			return compliance.inverse();
		}

		Eigen::Matrix<double, 6, 6> transversely_isotropic_stiffness_3d(const std::vector<double> &entries)
		{
			double Et = entries[0];
			double Ea = entries[1];
			double nu_t = entries[2];
			double nu_a = entries[3];
			double Ga = entries[4];

			Eigen::Matrix<double, 6, 6> compliance;
			compliance << 1.0 / Et, -nu_t / Et, -nu_a / Ea, 0.0, 0.0, 0.0,
				-nu_t / Et, 1.0 / Et, -nu_a / Ea, 0.0, 0.0, 0.0,
				-nu_a / Ea, -nu_a / Ea, 1.0 / Ea, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 1.0 / Ga, 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 1.0 / Ga, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, (2.0 * (1.0 + nu_t)) / Et;
			return compliance.inverse();
		}

		Eigen::Matrix<double, 6, 6> orthotropic_stiffness_3d(const std::vector<double> &entries)
		{
			double Ex = entries[0];
			double Ey = entries[1];
			double Ez = entries[2];
			double nuXY = entries[3];
			double nuXZ = entries[4];
			double nuYZ = entries[5];
			double muYZ = entries[6];
			double muZX = entries[7];
			double muXY = entries[8];

			double nuYX = nuXY * Ey / Ex;
			double nuZX = nuXZ * Ez / Ex;
			double nuZY = nuYZ * Ez / Ey;

			Eigen::Matrix<double, 6, 6> compliance;
			compliance << 1.0 / Ex, -nuYX / Ey, -nuZX / Ez, 0.0, 0.0, 0.0,
				-nuXY / Ex, 1.0 / Ey, -nuZY / Ez, 0.0, 0.0, 0.0,
				-nuXZ / Ex, -nuYZ / Ey, 1.0 / Ez, 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 1.0 / (2.0 * muYZ), 0.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 1.0 / (2.0 * muZX), 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / (2.0 * muXY);
			return compliance.inverse();
		}
	} // namespace

	template <>
	ElasticityTensor<double, 1> ElasticityTensor<double, 1>::from_lame(const LameParameter<double> &lame)
	{
		auto [lambda, mu] = lambda_mu<1>(lame);

		ElasticityTensor<double, 1> out;
		out.depends_on_elastic_coeff = true;
		out.stiffness(0, 0) = lambda + 2.0 * mu;
		return out;
	}

	template <>
	ElasticityTensor<double, 2> ElasticityTensor<double, 2>::from_lame(const LameParameter<double> &lame)
	{
		auto [lambda, mu] = lambda_mu<2>(lame);

		ElasticityTensor<double, 2> out;
		out.depends_on_elastic_coeff = true;
		double lambda_plus_2mu = lambda + 2.0 * mu;

		out.stiffness << lambda_plus_2mu, lambda, 0.0,
			lambda, lambda_plus_2mu, 0.0,
			0.0, 0.0, mu;

		return out;
	}

	template <>
	ElasticityTensor<double, 3> ElasticityTensor<double, 3>::from_lame(const LameParameter<double> &lame)
	{
		auto [lambda, mu] = lambda_mu<3>(lame);

		ElasticityTensor<double, 3> out;
		out.depends_on_elastic_coeff = true;
		double lambda_plus_2mu = lambda + 2.0 * mu;

		out.stiffness << lambda_plus_2mu, lambda, lambda, 0.0, 0.0, 0.0,
			lambda, lambda_plus_2mu, lambda, 0.0, 0.0, 0.0,
			lambda, lambda, lambda_plus_2mu, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, mu, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, mu, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, mu;

		return out;
	}

	template <typename Scalar, int dim>
	ElasticityTensor<utils::ExpressionValue, dim> ElasticityTensor<Scalar, dim>::from_json(
		const json &value,
		const Units &units,
		const std::string &root_path)
	{
		const std::string stress_unit = units.stress();

		if (!value.is_array())
		{
			throw std::runtime_error(fmt::format("elasticity_tensor: expected a JSON array"));
		}

		ElasticityTensor<utils::ExpressionValue, dim> out;
		out.depends_on_elastic_coeff = false;

		if constexpr (dim == 1)
		{
			// Direct stiffness in Voigt order [C00].
			parse_upper_symmetric_entries<1>(value, root_path, "elasticity_tensor", stress_unit, out.stiffness);
		}
		else if constexpr (dim == 2)
		{
			if (value.size() == 4)
			{
				// Orthotropic engineering constants [Ex, Ey, nuXY, muXY].
				build_expr_matrix_from_double<3>(orthotropic_stiffness_2d(numeric_entries(value, "elasticity_tensor")), stress_unit, out.stiffness);
			}
			else if (value.size() == 6)
			{
				// Upper-triangular stiffness in Voigt order [xx, yy, xy]:
				// [C00, C01, C02, C11, C12, C22].
				parse_upper_symmetric_entries<3>(value, root_path, "elasticity_tensor", stress_unit, out.stiffness);
			}
			else
			{
				throw std::runtime_error(fmt::format(
					"elasticity_tensor: expected 4 orthotropic entries or 6 stiffness entries for 2D, got {}",
					value.size()));
			}
		}
		else
		{
			if (value.size() == 5)
			{
				// Transversely-isotropic engineering constants [Et, Ea, nu_t, nu_a, Ga].
				build_expr_matrix_from_double<6>(transversely_isotropic_stiffness_3d(numeric_entries(value, "elasticity_tensor")), stress_unit, out.stiffness);
			}
			else if (value.size() == 9)
			{
				// Orthotropic engineering constants [Ex, Ey, Ez, nuXY, nuXZ, nuYZ, muYZ, muZX, muXY].
				build_expr_matrix_from_double<6>(orthotropic_stiffness_3d(numeric_entries(value, "elasticity_tensor")), stress_unit, out.stiffness);
			}
			else if (value.size() == 21)
			{
				// Upper-triangular stiffness in Voigt order [xx, yy, zz, yz, zx, xy].
				parse_upper_symmetric_entries<6>(value, root_path, "elasticity_tensor", stress_unit, out.stiffness);
			}
			else
			{
				throw std::runtime_error(fmt::format(
					"elasticity_tensor: expected 5 transverse-isotropic, 9 orthotropic, or 21 stiffness entries for 3D, got {}",
					value.size()));
			}
		}

		return out;
	}

	// template specialization
	template ElasticityTensor<utils::ExpressionValue, 1> ElasticityTensor<utils::ExpressionValue, 1>::from_json(const json &, const Units &, const std::string &);
	template ElasticityTensor<utils::ExpressionValue, 2> ElasticityTensor<utils::ExpressionValue, 2>::from_json(const json &, const Units &, const std::string &);
	template ElasticityTensor<utils::ExpressionValue, 3> ElasticityTensor<utils::ExpressionValue, 3>::from_json(const json &, const Units &, const std::string &);

	template <typename Scalar, int dim>
	ElasticityTensor<double, dim> ElasticityTensor<Scalar, dim>::eval_expr(double x, double y, double z, double t, int element_id) const
	{
		ElasticityTensor<double, dim> out{};
		// If depends on elastic coeff, we do not have enough info to compute
		// stiffness tensor here. User need to first evaluate lame parameters then
		// set tensor using from_lame factory.
		if (depends_on_elastic_coeff)
		{
			return out;
		}

		out.depends_on_elastic_coeff = false;
		for (int i = 0; i < VOIGT_SIZE; ++i)
		{
			for (int j = 0; j < VOIGT_SIZE; ++j)
			{
				out.stiffness(i, j) = stiffness(i, j)(x, y, z, t, element_id);
			}
		}
		return out;
	}

	template ElasticityTensor<double, 1> ElasticityTensor<utils::ExpressionValue, 1>::eval_expr(double, double, double, double, int) const;
	template ElasticityTensor<double, 2> ElasticityTensor<utils::ExpressionValue, 2>::eval_expr(double, double, double, double, int) const;
	template ElasticityTensor<double, 3> ElasticityTensor<utils::ExpressionValue, 3>::eval_expr(double, double, double, double, int) const;
} // namespace polyfem::material
