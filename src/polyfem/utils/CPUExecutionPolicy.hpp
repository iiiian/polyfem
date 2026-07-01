#pragma once

#include <thread>

namespace polyfem
{
	struct CPUExecutionPolicy
	{
		int thread_num;

		CPUExecutionPolicy()
		{
			int num = std::thread::hardware_concurrency();
			// 0 -> hardware thread undefined.
			thread_num = (num == 0) ? 1 : num;
		}
	};
} // namespace polyfem
