#include "FullNLProblem.hpp"
#include <polyfem/utils/Logger.hpp>

#include <algorithm>

namespace polyfem::solver
{
	namespace
	{
		std::string timing_key(const std::string &operation, const Form &form)
		{
			return operation + "/" + form.name();
		}

		std::vector<uint8_t> enabled_mask(const std::vector<std::shared_ptr<Form>> &forms)
		{
			std::vector<uint8_t> enabled;
			enabled.reserve(forms.size());
			for (const auto &form : forms)
				enabled.push_back(form->enabled() ? 1 : 0);
			return enabled;
		}
	} // namespace

	FullNLProblem::FullNLProblem(const std::vector<std::shared_ptr<Form>> &forms, ExecutionPolicy policy)
		: forms_(forms),
		  execution_policy_(policy)
	{
	}

	void FullNLProblem::finish()
	{
		log_timing_summary();

		for (auto &form : forms_)
			form->finish();
	}

	void FullNLProblem::log_timing_summary() const
	{
		if (!logger().should_log(spdlog::level::debug) || timings_.empty())
			return;

		std::vector<std::pair<std::string, utils::Timing>> sorted_timings(timings_.begin(), timings_.end());
		std::sort(sorted_timings.begin(), sorted_timings.end(), [](const auto &a, const auto &b) {
			return a.second.time > b.second.time;
		});

		double total_time = 0;
		for (const auto &[_, timing] : sorted_timings)
			total_time += timing.time;

		logger().debug("FullNLProblem timing summary: {:.6g}s accumulated over {} entries", total_time, sorted_timings.size());
		for (const auto &[name, timing] : sorted_timings)
		{
			const double percent = total_time > 0 ? 100 * timing.time / total_time : 0;
			const double average = timing.count > 0 ? timing.time / double(timing.count) : 0;
			logger().debug("{:48s}: {:10.6g}s {:5.1f}% ({:6} calls, avg {:10.6g}s)",
						   name, timing.time, percent, timing.count, average);
		}
	}

	double FullNLProblem::normalize_forms()
	{
		double total_weight = 0;
		for (const auto &f : forms_)
			total_weight += f->weight();

		logger().debug("Normalizing forms with scale: {}", total_weight);

		for (auto &f : forms_)
			f->set_scale(total_weight);

		return total_weight;
	}

	void FullNLProblem::init(const TVector &x)
	{
		for (auto &f : forms_)
			f->init(x);
	}

	void FullNLProblem::set_project_to_psd(bool project_to_psd)
	{
		for (auto &f : forms_)
			f->set_project_to_psd(project_to_psd);
	}

	void FullNLProblem::init_lagging(const TVector &x)
	{
		for (auto &f : forms_)
			f->init_lagging(x);
	}

	void FullNLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		for (auto &f : forms_)
			f->update_lagging(x, iter_num);
	}

	int FullNLProblem::max_lagging_iterations() const
	{
		int max_lagging_iterations = 1;
		for (auto &f : forms_)
			max_lagging_iterations = std::max(max_lagging_iterations, f->max_lagging_iterations());
		return max_lagging_iterations;
	}

	bool FullNLProblem::uses_lagging() const
	{
		for (auto &f : forms_)
			if (f->uses_lagging())
				return true;
		return false;
	}

	void FullNLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			f->line_search_begin(x0, x1);
	}

	void FullNLProblem::line_search_end()
	{
		for (auto &f : forms_)
			f->line_search_end();
	}

	double FullNLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		double step = 1;
		for (auto &f : forms_)
			if (f->enabled())
				step = std::min(step, f->max_step_size(x0, x1));
		return step;
	}

	bool FullNLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_valid(x0, x1))
				return false;
		return true;
	}

	bool FullNLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		for (auto &f : forms_)
			if (f->enabled() && !f->is_step_collision_free(x0, x1))
				return false;
		return true;
	}

	double FullNLProblem::value(const TVector &x)
	{
		double val = 0;
		for (auto &f : forms_)
			if (f->enabled())
			{
				utils::Timer timer(timings_[timing_key("value", *f)]);
				val += f->value_ng(x, execution_policy_);
			}
		return val;
	}

	void FullNLProblem::gradient(const TVector &x, TVector &grad)
	{
		DualVector grad_dual(x.size());
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			utils::Timer timer(timings_[timing_key("gradient", *f)]);
			f->first_derivative_ng(x, grad_dual, execution_policy_);
		}
		{
			utils::Timer timer(timings_["gradient/to_eigen"]);
			grad = grad_dual.to_eigen(execution_policy_);
		}
	}

	void FullNLProblem::hessian(const TVector &x, THessian &hessian)
	{
		const std::vector<uint8_t> enabled = enabled_mask(forms_);

		if (!hessian_bsr_
			|| hessian_bsr_ndof_ != x.size()
			|| hessian_bsr_enabled_ != enabled)
		{
			std::optional<BSRSparsityPattern> joined_pattern;

			for (auto &f : forms_)
			{
				if (!f->enabled())
					continue;

				utils::Timer timer(timings_[timing_key("hessian_sparsity", *f)]);
				auto pattern = f->hessian_sparsity_pattern_ng();
				if (!pattern)
					continue;

				if (pattern->rows != x.size() || pattern->cols != x.size())
					continue;

				if (joined_pattern)
					joined_pattern->join(*pattern);
				else
					joined_pattern = std::move(pattern);
			}

			if (joined_pattern)
				hessian_bsr_.emplace(*joined_pattern);
			else
				hessian_bsr_.emplace(x.size(), x.size());

			hessian_bsr_ndof_ = x.size();
			hessian_bsr_enabled_ = std::move(enabled);
		}

		{
			utils::Timer timer(timings_["hessian/reset_bsr"]);
			hessian_bsr_->reset(execution_policy_);
		}
		for (auto &f : forms_)
		{
			if (!f->enabled())
				continue;
			utils::Timer timer(timings_[timing_key("hessian", *f)]);
			f->second_derivative_ng(x, *hessian_bsr_, execution_policy_);
		}

		{
			utils::Timer timer(timings_["hessian/to_stiffness_matrix"]);
			hessian = hessian_bsr_->to_stiffness_matrix(execution_policy_);
		}
	}

	void FullNLProblem::solution_changed(const TVector &x)
	{
		for (auto &f : forms_)
			f->solution_changed(x);
	}

	void FullNLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		for (auto &f : forms_)
			f->post_step(data);
	}
} // namespace polyfem::solver
