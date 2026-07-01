#pragma once

#include <optional>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#endif

namespace polyfem
{

#ifdef POLYFEM_WITH_CUDA

	/// @brief Nullable device buffer.
	/// It's very annoying device_buffer does not have default ctor for empty buffer.
	template <typename T>
	using DBuf = std::optional<cuda::device_buffer<T>>;

#endif

} // namespace polyfem
