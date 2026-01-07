#include "SimplifiedNeoHookeanElasticityAutodiff.hpp"

namespace polyfem::assembler
{
	SimplifiedNeoHookeanAutodiff::SimplifiedNeoHookeanAutodiff()
	{
	}

	void SimplifiedNeoHookeanAutodiff::add_multimaterial(const int index, const json &params, const Units &units)
	{
		params_.add_multimaterial(index, params, size() == 3, units.stress());
	}
} // namespace polyfem::assembler
