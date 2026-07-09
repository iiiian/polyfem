#include "InertiaForm.hpp"

#include <polyfem/time_integrator/ImplicitTimeIntegrator.hpp>
#include <polyfem/utils/Types.hpp>

#include <cassert>

namespace polyfem::solver
{
	InertiaForm::InertiaForm(const StiffnessMatrix &mass,
							 const time_integrator::ImplicitTimeIntegrator &time_integrator)
		: mass_(mass), time_integrator_(time_integrator)
	{
		assert(mass.size() != 0);
	}

	double InertiaForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::VectorXd tmp = x - time_integrator_.x_tilde();
		// FIXME: DBC on x tilde
		const double prod = tmp.transpose() * mass_ * tmp;
		const double energy = 0.5 * prod;
		return energy;
	}

	void InertiaForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = mass_ * (x - time_integrator_.x_tilde());
	}

	std::optional<BSRSparsityPattern> InertiaForm::hessian_sparsity_pattern_ng() const
	{
		BSRSparsityPattern pattern{int(mass_.rows()), int(mass_.cols()), 1, {}};

		for (int k = 0; k < mass_.outerSize(); ++k)
		{
			for (StiffnessMatrix::InnerIterator it(mass_, k); it; ++it)
			{
				pattern.insert(it.row(), it.col());
			}
		}

		return pattern;
	}

	void InertiaForm::second_derivative_ng(const Eigen::VectorXd &x, BSRMatrix &hessian, ExecutionPolicy policy) const
	{
		(void)x;
		(void)policy;

		assert(mass_.rows() == hessian.rows());
		assert(mass_.cols() == hessian.cols());

		add_sparse_matrix_to_bsr_static(mass_, hessian.static_view(), weighted_scale());
	}

	void InertiaForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		hessian = mass_;
	}
} // namespace polyfem::solver
