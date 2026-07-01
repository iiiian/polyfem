#include <polyfem/utils/CUDAExecutionPolicy.hpp>

#include <cuda/stream>
#include <cuda/devices>
#include <cuda/memory_pool>

#include <optional>

namespace polyfem
{
	namespace
	{
		std::optional<cuda::device_ref> default_device;
		std::optional<cuda::stream> default_stream;
		std::optional<cuda::device_memory_pool> default_mem_pool;
		bool is_cuda_initialized = false;

		void default_init_cuda()
		{
			if (cuda::devices.size() == 0)
			{
				throw std::runtime_error("PolyFEM cuda init failed. No Nvidia GPU!!");
			}

			default_device.emplace(cuda::devices[0]);
			default_stream.emplace(*default_device);
			default_mem_pool.emplace(*default_device);
		}

		cuda::stream_ref get_default_stream()
		{
			if (!is_cuda_initialized)
			{
				default_init_cuda();
			}
			return *default_stream;
		}

		cuda::device_memory_pool_ref get_default_mr()
		{
			if (!is_cuda_initialized)
			{
				default_init_cuda();
			}
			return default_mem_pool->as_ref();
		}
	} // namespace

	// This is super hacky, not thread safe, and not customizable.
	// TODO: proper cuda init
	CudaExecutionPolicy::CudaExecutionPolicy() : stream(get_default_stream()), mr(get_default_mr())
	{
	}

}; // namespace polyfem
