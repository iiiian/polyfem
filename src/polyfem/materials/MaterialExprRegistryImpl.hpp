#pragma once

#include <cassert>
#include <tuple>
#include <type_traits>

#include <polyfem/utils/PackedArray.hpp>
#include <polyfem/utils/Span.hpp>
#include <polyfem/utils/ExpressionValue.hpp>

namespace polyfem::material
{

	template <template <typename> class... M>
	class MaterialExprRegistryImpl
	{
	private:
		using Expr = utils::ExpressionValue;

	public:
		MaterialExprRegistryImpl(int element_num) : element_num_(element_num) {};

		// -----------------------------------------------------
		// element APIs
		// -----------------------------------------------------

		int element_num() const { return element_num_; };

		bool has_element(int element) const { return (element < element_num_); }

		// -----------------------------------------------------
		// material APIs
		// -----------------------------------------------------

		/// @brief Return true if element has material expr T.
		template <typename T>
		bool has_material(int element) const
		{
			static_assert((std::is_same_v<T, M<Expr>> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return (c.get(element) != nullptr);
		}

		/// @brief Return true if element has material T.
		template <template <typename> class T>
		bool has_material(int element) const
		{
			return has_material<T<Expr>>(element);
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		template <typename T>
		T *get(int element)
		{
			static_assert((std::is_same_v<T, M<Expr>> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return c.get(element);
		}

		/// @brief Return ptr to element material T. nullptr if not exists.
		template <template <typename> class T>
		T<Expr> *get(int element)
		{
			return get<T<Expr>>(element);
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		template <typename T>
		const T *get(int element) const
		{
			static_assert((std::is_same_v<T, M<Expr>> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return c.get(element);
		}

		/// @brief Return ptr to element material T. nullptr if not exists.
		template <template <typename> class T>
		const T<Expr> *get(int element) const
		{
			return get<T<Expr>>(element);
		}

		// /// @brief Return all materials of type T.
		// template <template <typename> class T>
		// Span<T<Expr>> get_all_materials()
		// {
		// 	static_assert((std::is_same_v<T<Expr>, M<Expr>> || ...),
		// 				  "T is not a material type, double check T appears as "
		// 				  "template argument in registry declaration.");
		//
		// 	auto &c = std::get<utils::PackedArray<T<Expr>>>(materials_);
		// 	return c.data();
		// }
		//
		// /// @brief Return all materials of type T.
		// template <template <typename> class T>
		// Span<const T<Expr>> get_all_materials() const
		// {
		// 	static_assert((std::is_same_v<T<Expr>, M<Expr>> || ...),
		// 				  "T is not a material type, double check T appears as "
		// 				  "template argument in registry declaration.");
		//
		// 	auto &c = std::get<utils::PackedArray<T<Expr>>>(materials_);
		// 	return c.data();
		// }

		/// @brief Remove material expr T. No-op if element does not have such material.
		template <typename T>
		void remove(int element)
		{
			static_assert((std::is_same_v<T, M<Expr>> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			c.remove(element);
		}

		/// @brief Remove material T. No-op if element does not have such material.
		template <template <typename> class T>
		void remove(int element)
		{
			remove<T<Expr>>(element);
		}

		// /// @brief Remove material T. No-op if element does not have such material.
		// template <template <typename> class T>
		// void remove_all_materials()
		// {
		// 	static_assert((std::is_same_v<T<Expr>, M<Expr>> || ...),
		// 				  "T is not a material type, double check T appears as "
		// 				  "template argument in registry declaration.");
		//
		// 	auto &c = std::get<utils::PackedArray<T<Expr>>>(materials_);
		// 	c.clear();
		// }

		/// @brief Set/Replace material expr T of element.
		/// @return Pointer to newly modified material. nullptr if element is invalid.
		template <typename T>
		T *set(int element, T material)
		{
			static_assert((std::is_same_v<T, M<Expr>> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");
			if (!has_element(element))
			{
				return nullptr;
			}

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return c.set(element, std::move(material));
		}

		/// @brief Set/Replace material T of element.
		/// @return Pointer to newly modified material. nullptr if element is invalid.
		template <template <typename> class T>
		T<Expr> *set(int element, T<Expr> material)
		{
			return set(element, material);
		}

	private:
		int element_num_;
		std::tuple<utils::PackedArray<M<Expr>>...> materials_;
	};

} // namespace polyfem::material
