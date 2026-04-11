#pragma once

#ifdef POLYFEM_WITH_CUDA

#include <cuda/std/span>

namespace polyfem
{
	template <typename T, size_t extent = cuda::std::dynamic_extent>
	using Span = cuda::std::span<T, extent>;
}

#else

#include <nonstd/span.hpp>

namespace polyfem
{
	template <typename T, nonstd::span_lite::extent_t extent = nonstd::dynamic_extent>
	using Span = nonstd::span<T, extent>;
}

#endif
