#include "polyfem/utils/span.hpp"
#include <cassert>

namespace polyfem
{
	struct Range
	{
		int offset = -1;
		int num = -1;

		operator bool() const
		{
			return (offset >= 0 && num >= 0);
		}
	};

	template <typename T>
	span<T> slice_by_range(span<T> s, Range r)
	{
		if (!r)
		{
			return {};
		}
		assert(r.offset >= 0 && r.num >= 0);
		assert(r.offset + r.num <= s.size());
		return s.subspan(r.offset, r.num);
	}

} // namespace polyfem
