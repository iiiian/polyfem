#pragma once

#ifdef POLYFEM_WITH_CUDA

#include <cuda/stream>
#include <cuda/memory_resource>

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
	cu::mr::resource_ref<cu::mr::device_accessible> mr;
};

#else

#define __both__

#endif
