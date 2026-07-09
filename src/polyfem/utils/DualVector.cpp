#include <polyfem/utils/DualVector.hpp>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/algorithm>
#endif

namespace polyfem
{
	DualVector::DualVector(int size)
	{
		host_values_.resize(size, 0.0);
	}

	Span<double> DualVector::host_view()
	{
		return host_values_;
	}

	Span<const double> DualVector::host_view() const
	{
		return host_values_;
	}

	Eigen::VectorXd DualVector::to_eigen(ExecutionPolicy policy) const
	{
		Eigen::VectorXd out(size());
		for (int i = 0; i < size(); ++i)
			out[i] = host_values_[i];

#ifdef POLYFEM_WITH_CUDA
		// Sum device values.
		if (policy.mode == ExecutionMode::Hybrid && device_values_)
		{
			std::vector<double> tmp(size());
			cuda::copy_bytes(*policy.stream, *device_values_, tmp);
			policy.stream->sync();
			for (int i = 0; i < size(); ++i)
				out[i] += tmp[i];
		}
#else
		(void)policy;
#endif

		return out;
	}

#ifdef POLYFEM_WITH_CUDA
	Span<double> DualVector::device_view(ExecutionPolicy policy)
	{
		assert(policy.mode == ExecutionMode::Hybrid);

		if (!device_values_)
		{
			device_values_ =
				cuda::make_buffer<double>(*policy.stream, *policy.mr, host_values_.size(), cuda::no_init);
			cuda::fill_bytes(*policy.stream, *device_values_, 0);
			policy.stream->sync();
		}

		return *device_values_;
	}
#endif

} // namespace polyfem
