#pragma once

#include <polyfem/assembler/AssemblyEssentials.hpp>

namespace polyfem::assembler
{
	using ElementBasesView [[deprecated("Use AssemblyEssentialsView")]] = AssemblyEssentialsView;
	using ElementBases [[deprecated("Use AssemblyEssentials")]] = AssemblyEssentials;
} // namespace polyfem::assembler
