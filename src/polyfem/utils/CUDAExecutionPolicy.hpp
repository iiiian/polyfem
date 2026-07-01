#pragma once

#include <cuda/stream>
#include <cuda/memory_pool>

namespace polyfem
{

	struct CudaExecutionPolicy
	{
		cuda::stream_ref stream;
		cuda::device_memory_pool_ref mr;

		CudaExecutionPolicy();
	};
}; // namespace polyfem
