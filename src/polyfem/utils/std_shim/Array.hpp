#pragma once

#ifdef POLYFEM_WITH_CUDA

#include <cuda/std/array>

namespace polyfem
{
	template <typename T, size_t size>
	using Array = cuda::std::array<T, size>;
}

#else

#include <array>

namespace polyfem
{
	template <typename T, size_t size>
	using Array = std::array<T, size>;
}

#endif
