#include <polyfem/utils/vm/Program.hpp>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#endif

#include <polyfem/utils/vm/tinyexpr.h>
#include <polyfem/utils/CudaUtils.hpp>
#include <polyfem/utils/CpuPolicy.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/std_shim/Optional.hpp>

#include <vector>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/algorithm>
#endif

namespace polyfem::utils::vm
{
	namespace
	{
		template <OpCode code>
		void dummy() {}

		Optional<OpCode> decode_func_ptr(const void *ptr)
		{
			if (ptr == (const void *)get_add_func_ptr())
				return OpCode::ADD;
			else if (ptr == (const void *)get_sub_func_ptr())
				return OpCode::SUB;
			else if (ptr == (const void *)get_mul_func_ptr())
				return OpCode::MUL;
			else if (ptr == (const void *)get_div_func_ptr())
				return OpCode::DIV;
			else if (ptr == (const void *)get_neg_func_ptr())
				return OpCode::NEG;
			else if (ptr == (const void *)get_comma_func_ptr())
				return OpCode::COMMA;
			else if (ptr == (const void *)get_fac_func_ptr())
				return OpCode::FAC;
			else if (ptr == (const void *)get_ncr_func_ptr())
				return OpCode::NCR;
			else if (ptr == (const void *)get_npr_func_ptr())
				return OpCode::NPR;
			else if (ptr == (const void *)fabs)
				return OpCode::ABS;
			else if (ptr == (const void *)acos)
				return OpCode::ACOS;
			else if (ptr == (const void *)asin)
				return OpCode::ASIN;
			else if (ptr == (const void *)atan)
				return OpCode::ATAN;
			else if (ptr == (const void *)atan2)
				return OpCode::ATAN2;
			else if (ptr == (const void *)ceil)
				return OpCode::CEIL;
			else if (ptr == (const void *)cos)
				return OpCode::COS;
			else if (ptr == (const void *)cosh)
				return OpCode::COSH;
			else if (ptr == (const void *)exp)
				return OpCode::EXP;
			else if (ptr == (const void *)floor)
				return OpCode::FLOOR;
			else if (ptr == (const void *)log)
				return OpCode::LOG;
			else if (ptr == (const void *)log10)
				return OpCode::LOG10;
			else if (ptr == (const void *)pow)
				return OpCode::POW;
			else if (ptr == (const void *)sin)
				return OpCode::SIN;
			else if (ptr == (const void *)sinh)
				return OpCode::SINH;
			else if (ptr == (const void *)sqrt)
				return OpCode::SQRT;
			else if (ptr == (const void *)tan)
				return OpCode::TAN;
			else if (ptr == (const void *)tanh)
				return OpCode::TANH;
			else if (ptr == (const void *)dummy<OpCode::MIN>)
				return OpCode::MIN;
			else if (ptr == (const void *)dummy<OpCode::MAX>)
				return OpCode::MAX;
			else if (ptr == (const void *)dummy<OpCode::SMOOTH_STEP>)
				return OpCode::SMOOTH_STEP;
			else if (ptr == (const void *)dummy<OpCode::HALF_SMOOTH_STEP>)
				return OpCode::HALF_SMOOTH_STEP;
			else if (ptr == (const void *)dummy<OpCode::DEG2GRAD>)
				return OpCode::DEG2GRAD;
			else if (ptr == (const void *)dummy<OpCode::ROTATE_2D_X>)
				return OpCode::ROTATE_2D_X;
			else if (ptr == (const void *)dummy<OpCode::ROTATE_2D_Y>)
				return OpCode::ROTATE_2D_Y;
			else if (ptr == (const void *)dummy<OpCode::ROTATE_2D_Z>)
				return OpCode::ROTATE_2D_Z;
			else if (ptr == (const void *)dummy<OpCode::SMOOTH_ABS>)
				return OpCode::SMOOTH_ABS;
			else if (ptr == (const void *)dummy<OpCode::IF_LARGER_THAN_ZERO_THEN_ELSE>)
				return OpCode::IF_LARGER_THAN_ZERO_THEN_ELSE;
			else if (ptr == (const void *)dummy<OpCode::SIGN>)
				return OpCode::SIGN;
			else if (ptr == (const void *)dummy<OpCode::COMPARE>)
				return OpCode::COMPARE;

			return nullopt;
		}

		enum class TENodeType
		{
			Variable,
			Constant,
			Function,
			Closure,
			Unknown
		};

		struct TENodeInfo
		{
			TENodeType type;
			int arity; // Meaningful only when type is function or closure.

			TENodeInfo(int type_flag)
			{
				const int base = type_flag & 0x1F;
				if (base == TE_VARIABLE)
				{
					type = TENodeType::Variable;
				}
				else if (base == 1) // TE_CONSTANT
				{
					type = TENodeType::Constant;
				}
				else if ((type_flag & TE_FUNCTION0) != 0)
				{
					type = TENodeType::Function;
				}
				else if ((type_flag & TE_CLOSURE0) != 0)
				{
					type = TENodeType::Closure;
				}
				else
				{
					type = TENodeType::Unknown;
				}

				arity = type_flag & 0x07;
			}
		};

		enum class CompilerError
		{
			ILLEGAL_EXPR,
			EXCEED_MAX_STACK,
			EXCEED_MAX_CONST,
		};

		// return new stack idx.
		int parse_ast(const te_expr *node,
					  const double *var_base,
					  int stack_idx,
					  std::vector<double> &constants,
					  std::vector<Instruction> &instructions)
		{
			assert(node);
			assert(stack_idx >= 0);

			if (stack_idx > MAX_STACK_IDX)
			{
				throw CompilerError::EXCEED_MAX_STACK;
			}

			TENodeInfo info{node->type};
			switch (info.type)
			{
			case TENodeType::Constant:
			{
				size_t offset = constants.size();
				if (offset > MAX_CONST_IDX)
				{
					throw CompilerError::EXCEED_MAX_CONST;
				}
				constants.push_back(node->value);
				instructions.push_back({OpCode::LOAD_CONST, static_cast<uint16_t>(offset)});
				return stack_idx + 1;
			}
			case TENodeType::Variable:
			{
				int offset = node->bound - var_base;
				// We only have 4 variables x,y,z,t
				assert(offset >= 0 && offset < 4);
				instructions.push_back({OpCode::LOAD_VAR, static_cast<uint16_t>(offset)});
				return stack_idx + 1;
			}
			case TENodeType::Function:
			{
				assert(info.arity >= 0 && info.arity <= 7);
				for (int i = 0; i < info.arity; ++i)
				{
					stack_idx = parse_ast(static_cast<te_expr *>(node->parameters[i]), var_base, stack_idx, constants, instructions);
				}

				auto code = decode_func_ptr(node->function);
				assert(code.has_value());
				instructions.push_back({*code, 0});
				return stack_idx - info.arity + 1;
			}
			case TENodeType::Closure:
			{
				assert(false && "TE closure nodes are not supported by Program");
				return stack_idx;
			}
			default:
			{
				assert(false && "Unknown TE node type");
				return stack_idx;
			}
			}
		}

		void compile_expr(const char *expr, std::vector<double> &constants, std::vector<Instruction> instructions)
		{
			double var[4];
			std::vector<te_variable> vars = {
				{"x", var, TE_VARIABLE},
				{"y", var + 1, TE_VARIABLE},
				{"z", var + 2, TE_VARIABLE},
				{"t", var + 3, TE_VARIABLE},
				{"min", (const void *)dummy<OpCode::MIN>, TE_FUNCTION2},
				{"max", (const void *)dummy<OpCode::MAX>, TE_FUNCTION2},
				{"smoothstep", (const void *)dummy<OpCode::SMOOTH_STEP>, TE_FUNCTION1},
				{"half_smoothstep", (const void *)dummy<OpCode::HALF_SMOOTH_STEP>, TE_FUNCTION1},
				{"deg2rad", (const void *)dummy<OpCode::DEG2GRAD>, TE_FUNCTION1},
				{"rotate_2D_x", (const void *)dummy<OpCode::ROTATE_2D_X>, TE_FUNCTION3},
				{"rotate_2D_y", (const void *)dummy<OpCode::ROTATE_2D_Y>, TE_FUNCTION3},
				{"rotate_2D_z", (const void *)dummy<OpCode::ROTATE_2D_Z>, TE_FUNCTION3},
				{"if", (const void *)dummy<OpCode::IF_LARGER_THAN_ZERO_THEN_ELSE>, TE_FUNCTION3},
				{"compare", (const void *)dummy<OpCode::COMPARE>, TE_FUNCTION2},
				{"smooth_abs", (const void *)dummy<OpCode::SMOOTH_ABS>, TE_FUNCTION2},
				{"sign", (const void *)dummy<OpCode::SIGN>, TE_FUNCTION1},
			};

			int err;
			te_expr *ast = te_compile(expr, vars.data(), vars.size(), &err);
			if (!ast)
			{
				throw CompilerError::ILLEGAL_EXPR;
			}

			constants.clear();
			instructions.clear();

			int stack_idx = parse_ast(ast, var, 0, constants, instructions);
			assert(stack_idx == 1);

			te_free(ast);
		}

	} // namespace

	template <>
	Program<CpuPolicy>::Program(const char *expr, CpuPolicy policy)
	{
		try
		{
			compile_expr(expr, h_constants, h_instructions);
		}
		catch (CompilerError err)
		{
			if (err == CompilerError::ILLEGAL_EXPR)
			{
				log_and_throw_error("Illegal Expression :\n{} \n", expr);
			}
			else if (err == CompilerError::EXCEED_MAX_STACK)
			{
				log_and_throw_error("Expression is too complex:\n{} \nReason: Exceed maximum stack size. "
									"If you do have a use case for very complex expression string, please file a issue.",
									expr);
			}
			else
			{
				log_and_throw_error("Expression is too complex.\n{} \nReason: Exceed maximum constant size. "
									"If you do have a use case for very complex expression string, please file a issue.",
									expr);
			}
		}
	}

	template <>
	ProgramView Program<CpuPolicy>::view() const
	{
		return ProgramView{h_constants, h_instructions};
	}

#ifdef POLYFEM_WITH_CUDA

	template <>
	Program<CudaPolicy>::Program(const char *expr, CudaPolicy policy)
	{
		try
		{
			compile_expr(expr, h_constants, h_instructions);
		}
		catch (CompilerError err)
		{
			if (err == CompilerError::ILLEGAL_EXPR)
			{
				log_and_throw_error("Illegal Expression :\n{} \n", expr);
			}
			else if (err == CompilerError::EXCEED_MAX_STACK)
			{
				log_and_throw_error("Expression is too complex:\n{} \nReason: Exceed maximum stack size. "
									"If you do have a use case for very complex expression string, please file a issue.",
									expr);
			}
			else
			{
				log_and_throw_error("Expression is too complex.\n{} \nReason: Exceed maximum constant size. "
									"If you do have a use case for very complex expression string, please file a issue.",
									expr);
			}
		}

		d_constants = cu::make_buffer<double>(policy.stream, policy.mr, h_constants.size(), cu::no_init);
		cu::copy_bytes(policy.stream, h_constants, *d_constants);
		h_constants.clear();

		d_instructions = cu::make_buffer<Instruction>(policy.stream, policy.mr, h_instructions.size(), cu::no_init);
		cu::copy_bytes(policy.stream, h_instructions, *d_instructions);
		h_instructions.clear();
	}

	template <>
	ProgramView Program<CudaPolicy>::view() const
	{
		return ProgramView{*d_constants, *d_instructions};
	}
#endif

} // namespace polyfem::utils::vm
