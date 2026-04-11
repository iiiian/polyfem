#pragma once

#ifdef POLYFEM_WITH_CUDA

#include <cuda/std/optional>

namespace polyfem
{
	template <typename T>
	using Optional = cuda::std::optional<T>;

	constexpr auto nullopt = cuda::std::nullopt;
} // namespace polyfem

#else

#include <optional>

namespace polyfem
{
	template <typename T>
	using Optional = std::optional<T>;

	constexpr auto nullopt = std::nullopt;
} // namespace polyfem

#endif
