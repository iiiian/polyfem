#pragma once

#include <optional>
#include <cuda/buffer>

namespace polyfem
{

	/// @brief Nullable device buffer.
	/// It's very annoying device_buffer does not have default ctor for empty buffer.
	template <typename T>
	using DeviceBuf = std::optional<cuda::device_buffer<T>>;

} // namespace polyfem
