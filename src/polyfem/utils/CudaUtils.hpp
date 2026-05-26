#pragma once

#ifdef POLYFEM_WITH_CUDA

#include <cuda/stream>
#include <cuda/memory_pool>

#define __both__ __host__ __device__

namespace cuda
{
}
namespace cuda::std
{
}

namespace cu = cuda;
namespace ctd = cuda::std;

struct CudaPolicy
{
	cu::stream_ref stream;
	cu::device_memory_pool_ref mr;
};

#else

#define __both__

#endif
