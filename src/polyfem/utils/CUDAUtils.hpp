#pragma once

#include <polyfem/utils/CudaBoth.hpp>

#include <optional>
#include <cuda/buffer>

namespace polyfem
{
	POLYFEM_BOTH constexpr int div_round_up(int n, int d)
	{
		return (n + d - 1) / d;
	}

	/// @brief Nullable device buffer.
	/// It's very annoying device_buffer does not have default ctor for empty buffer.
	template <typename T>
	using DeviceBuf = std::optional<cuda::device_buffer<T>>;

} // namespace polyfem
