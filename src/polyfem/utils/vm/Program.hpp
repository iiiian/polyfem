#pragma once

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#endif

// MSVC need this macro for M_PI
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>

#include <polyfem/utils/std_shim/Span.hpp>
#include <polyfem/utils/std_shim/Optional.hpp>
#include <polyfem/utils/CudaUtils.hpp>
#include <polyfem/utils/CpuPolicy.hpp>

namespace polyfem::utils::vm
{

	constexpr int MAX_CONST_IDX = std::numeric_limits<uint16_t>::max();
	// Unless you are nesting expression like crazy, 63 should be enough.
	constexpr int MAX_STACK_IDX = 63;

	enum class OpCode : uint8_t
	{
		// Aux
		LOAD_CONST,
		LOAD_VAR,

		// Tinyexpr builtin
		ADD,
		SUB,
		MUL,
		DIV,
		NEG,
		COMMA,
		FAC,
		NCR,
		NPR,
		// mat.h builtin
		ABS,
		ACOS,
		ASIN,
		ATAN,
		ATAN2,
		CEIL,
		COS,
		COSH,
		EXP,
		FLOOR,
		LN,
		LOG,
		LOG10,
		POW,
		SIN,
		SINH,
		SQRT,
		TAN,
		TANH,

		// PolyFEM custom
		MIN,
		MAX,
		SMOOTH_STEP,
		HALF_SMOOTH_STEP,
		DEG2GRAD,
		ROTATE_2D_X,
		ROTATE_2D_Y,
		ROTATE_2D_Z,
		SMOOTH_ABS,
		IF_LARGER_THAN_ZERO_THEN_ELSE,
		SIGN,
		COMPARE
	};

	struct Instruction
	{
		OpCode code;
		uint16_t operand;
	};

	struct ProgramView
	{
		Span<const double> constants;
		Span<const Instruction> instructions;

		__both__ double eval(double x, double y, double z, double t) const
		{
			double var[4];
			var[0] = x;
			var[1] = y;
			var[2] = z;
			var[3] = t;

			double st[MAX_STACK_IDX + 1];
			int ip = 0; // instruction ptr

			for (Instruction ins : instructions)
			{
				OpCode code = ins.code;
				switch (code)
				{
				case OpCode::LOAD_CONST:
					st[ip++] = constants[ins.operand];
					break;
				case OpCode::LOAD_VAR:
					st[ip++] = var[ins.operand];
					break;
				case OpCode::ADD:
					st[ip - 2] = st[ip - 2] + st[ip - 1];
					--ip;
					break;
				case OpCode::SUB:
					st[ip - 2] = st[ip - 2] - st[ip - 1];
					--ip;
					break;
				case OpCode::MUL:
					st[ip - 2] = st[ip - 2] * st[ip - 1];
					--ip;
					break;
				case OpCode::DIV:
					st[ip - 2] = st[ip - 2] / st[ip - 1];
					--ip;
					break;
				case OpCode::NEG:
					st[ip - 1] = -st[ip - 1];
					break;
				case OpCode::COMMA:
					st[ip - 2] = st[ip - 1];
					--ip;
					break;
				case OpCode::FAC:
				{
					double a = st[ip - 1];
					if (a < 0.0)
					{
						st[ip - 1] = std::numeric_limits<double>::quiet_NaN();
					}
					else if (a > std::numeric_limits<uint32_t>::max())
					{
						st[ip - 1] = std::numeric_limits<double>::infinity();
					}
					else
					{
						uint32_t ua = static_cast<uint32_t>(a);
						uint64_t result = 1;
						for (uint64_t i = 1; i <= ua; ++i)
						{
							if (i > std::numeric_limits<uint64_t>::max() / result)
							{
								result = 0;
								st[ip - 1] = std::numeric_limits<double>::infinity();
								break;
							}
							result *= i;
						}
						if (result != 0)
							st[ip - 1] = static_cast<double>(result);
					}
					break;
				}
				case OpCode::NCR:
				{
					double n = st[ip - 2];
					double r = st[ip - 1];
					if (n < 0.0 || r < 0.0 || n < r)
					{
						st[ip - 2] = std::numeric_limits<double>::quiet_NaN();
					}
					else if (n > std::numeric_limits<uint32_t>::max() || r > std::numeric_limits<uint32_t>::max())
					{
						st[ip - 2] = std::numeric_limits<double>::infinity();
					}
					else
					{
						uint64_t un = static_cast<uint32_t>(n);
						uint64_t ur = static_cast<uint32_t>(r);
						uint64_t result = 1;
						if (ur > un / 2)
							ur = un - ur;
						for (uint64_t i = 1; i <= ur; ++i)
						{
							if (result > std::numeric_limits<uint64_t>::max() / (un - ur + i))
							{
								result = 0;
								st[ip - 2] = std::numeric_limits<double>::infinity();
								break;
							}
							result *= un - ur + i;
							result /= i;
						}
						if (result != 0)
							st[ip - 2] = static_cast<double>(result);
					}
					--ip;
					break;
				}
				case OpCode::NPR:
				{
					double n = st[ip - 2];
					double r = st[ip - 1];
					if (n < 0.0 || r < 0.0 || n < r)
					{
						st[ip - 2] = std::numeric_limits<double>::quiet_NaN();
					}
					else if (n > std::numeric_limits<uint32_t>::max() || r > std::numeric_limits<uint32_t>::max())
					{
						st[ip - 2] = std::numeric_limits<double>::infinity();
					}
					else
					{
						uint64_t un = static_cast<uint32_t>(n);
						uint64_t ur = static_cast<uint32_t>(r);
						uint64_t result = 1;
						if (ur > un / 2)
							ur = un - ur;
						for (uint64_t i = 1; i <= ur; ++i)
						{
							if (result > std::numeric_limits<uint64_t>::max() / (un - ur + i))
							{
								result = 0;
								st[ip - 2] = std::numeric_limits<double>::infinity();
								break;
							}
							result *= un - ur + i;
							result /= i;
						}
						if (result != 0)
						{
							if (r < 0.0)
							{
								st[ip - 2] = std::numeric_limits<double>::quiet_NaN();
							}
							else if (r > std::numeric_limits<uint32_t>::max())
							{
								st[ip - 2] = std::numeric_limits<double>::infinity();
							}
							else
							{
								uint64_t fac_result = 1;
								uint32_t ur_fac = static_cast<uint32_t>(r);
								bool fac_overflow = false;
								for (uint64_t i = 1; i <= ur_fac; ++i)
								{
									if (i > std::numeric_limits<uint64_t>::max() / fac_result)
									{
										fac_overflow = true;
										st[ip - 2] = std::numeric_limits<double>::infinity();
										break;
									}
									fac_result *= i;
								}
								if (!fac_overflow)
									st[ip - 2] = static_cast<double>(result) * static_cast<double>(fac_result);
							}
						}
					}
					--ip;
					break;
				}
				case OpCode::ABS:
					st[ip - 1] = std::fabs(st[ip - 1]);
					break;
				case OpCode::ACOS:
					st[ip - 1] = std::acos(st[ip - 1]);
					break;
				case OpCode::ASIN:
					st[ip - 1] = std::asin(st[ip - 1]);
					break;
				case OpCode::ATAN:
					st[ip - 1] = std::atan(st[ip - 1]);
					break;
				case OpCode::ATAN2:
					st[ip - 2] = std::atan2(st[ip - 2], st[ip - 1]);
					--ip;
					break;
				case OpCode::CEIL:
					st[ip - 1] = std::ceil(st[ip - 1]);
					break;
				case OpCode::COS:
					st[ip - 1] = std::cos(st[ip - 1]);
					break;
				case OpCode::COSH:
					st[ip - 1] = std::cosh(st[ip - 1]);
					break;
				case OpCode::EXP:
					st[ip - 1] = std::exp(st[ip - 1]);
					break;
				case OpCode::FLOOR:
					st[ip - 1] = std::floor(st[ip - 1]);
					break;
				case OpCode::LN:
					st[ip - 1] = std::log(st[ip - 1]);
					break;
				case OpCode::LOG:
					st[ip - 1] = std::log(st[ip - 1]);
					break;
				case OpCode::LOG10:
					st[ip - 1] = std::log10(st[ip - 1]);
					break;
				case OpCode::POW:
					st[ip - 2] = std::pow(st[ip - 2], st[ip - 1]);
					--ip;
					break;
				case OpCode::SIN:
					st[ip - 1] = std::sin(st[ip - 1]);
					break;
				case OpCode::SINH:
					st[ip - 1] = std::sinh(st[ip - 1]);
					break;
				case OpCode::SQRT:
					st[ip - 1] = std::sqrt(st[ip - 1]);
					break;
				case OpCode::TAN:
					st[ip - 1] = std::tan(st[ip - 1]);
					break;
				case OpCode::TANH:
					st[ip - 1] = std::tanh(st[ip - 1]);
					break;
				case OpCode::MIN:
					st[ip - 2] = st[ip - 2] < st[ip - 1] ? st[ip - 2] : st[ip - 1];
					--ip;
					break;
				case OpCode::MAX:
					st[ip - 2] = st[ip - 2] > st[ip - 1] ? st[ip - 2] : st[ip - 1];
					--ip;
					break;
				case OpCode::SMOOTH_STEP:
					if (st[ip - 1] < 0)
						st[ip - 1] = 0;
					else if (st[ip - 1] > 1)
						st[ip - 1] = 1;
					else
						st[ip - 1] = 3 * std::pow(st[ip - 1], 2) - 2 * std::pow(st[ip - 1], 3);
					break;
				case OpCode::HALF_SMOOTH_STEP:
				{
					double b = (st[ip - 1] + 1.) / 2.;
					if (b < 0)
						st[ip - 1] = -1;
					else if (b > 1)
						st[ip - 1] = 1;
					else
						st[ip - 1] = 2. * (3 * std::pow(b, 2) - 2 * std::pow(b, 3)) - 1.;
					break;
				}
				case OpCode::DEG2GRAD:
					st[ip - 1] = st[ip - 1] * M_PI / 180.0;
					break;
				case OpCode::ROTATE_2D_X:
				{
					double x0 = st[ip - 3];
					double y0 = st[ip - 2];
					double theta = st[ip - 1];
					st[ip - 3] = x0 * std::cos(theta) - y0 * std::sin(theta);
					ip -= 2;
					break;
				}
				case OpCode::ROTATE_2D_Y:
				{
					double x0 = st[ip - 3];
					double y0 = st[ip - 2];
					double theta = st[ip - 1];
					st[ip - 3] = x0 * std::sin(theta) + y0 * std::cos(theta);
					ip -= 2;
					break;
				}
				case OpCode::ROTATE_2D_Z:
				{
					// TODO: probably a bug.
					st[ip - 3] = 0.0;
					ip -= 2;
					break;
				}
				case OpCode::SMOOTH_ABS:
				{
					double x0 = st[ip - 2];
					double k = st[ip - 1];
					st[ip - 2] = std::tanh(k * x0) * x0;
					--ip;
					break;
				}
				case OpCode::IF_LARGER_THAN_ZERO_THEN_ELSE:
				{
					double check = st[ip - 3];
					double ttrue = st[ip - 2];
					double ffalse = st[ip - 1];
					st[ip - 3] = check >= 0 ? ttrue : ffalse;
					ip -= 2;
					break;
				}
				case OpCode::SIGN:
				{
					double x0 = st[ip - 1];
					st[ip - 1] = (0 < x0) - (x0 < 0);
					break;
				}
				case OpCode::COMPARE:
				{
					double a = st[ip - 2];
					double b = st[ip - 1];
					st[ip - 2] = a < b ? 1.0 : 0.0;
					--ip;
					break;
				}
				default:
					assert(false && "Unknown OpCode");
				}
			}

			assert(ip == 1);
			return st[0];
		}
	};

	template <typename ExecutionPolicy = CpuPolicy>
	class Program
	{
	public:
		// std::string is not compatible with cuda, so fallback to C style string.
		Program(const char *expr, ExecutionPolicy policy = {});

		ProgramView view() const;

	private:
#ifdef POLYFEM_WITH_CUDA
		Optional<cu::device_buffer<double>> d_constants;
		Optional<cu::device_buffer<Instruction>> d_instructions;
#endif

		std::vector<double> h_constants;
		std::vector<Instruction> h_instructions;
	};

} // namespace polyfem::utils::vm
