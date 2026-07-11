#pragma once

#include <tuple>
#include <type_traits>

#include <polyfem/utils/PackedArray.hpp>

namespace polyfem::material
{

	template <typename... M>
	class MaterialExprRegistryImpl
	{
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
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return (c.get(element) != nullptr);
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		template <typename T>
		T *get(int element)
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return c.get(element);
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		template <typename T>
		const T *get(int element) const
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return c.get(element);
		}

		// /// @brief Return all materials of type T.
		// template <typename T>
		// Span<T> get_all_materials()
		// {
		// 	static_assert((std::is_same_v<T, M> || ...),
		// 				  "T is not a material type, double check T appears as "
		// 				  "template argument in registry declaration.");
		//
		// 	auto &c = std::get<utils::PackedArray<T>>(materials_);
		// 	return c.data();
		// }
		//
		// /// @brief Return all materials of type T.
		// template <typename T>
		// Span<const T> get_all_materials() const
		// {
		// 	static_assert((std::is_same_v<T, M> || ...),
		// 				  "T is not a material type, double check T appears as "
		// 				  "template argument in registry declaration.");
		//
		// 	auto &c = std::get<utils::PackedArray<T>>(materials_);
		// 	return c.data();
		// }

		/// @brief Remove material expr T. No-op if element does not have such material.
		template <typename T>
		void remove(int element)
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			c.remove(element);
		}

		// /// @brief Remove material T. No-op if element does not have such material.
		// template <typename T>
		// void remove_all_materials()
		// {
		// 	static_assert((std::is_same_v<T, M> || ...),
		// 				  "T is not a material type, double check T appears as "
		// 				  "template argument in registry declaration.");
		//
		// 	auto &c = std::get<utils::PackedArray<T>>(materials_);
		// 	c.clear();
		// }

		/// @brief Set/Replace material expr T of element.
		/// @return Pointer to newly modified material. nullptr if element is invalid.
		template <typename T>
		T *set(int element, T material)
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");
			if (!has_element(element))
			{
				return nullptr;
			}

			auto &c = std::get<utils::PackedArray<T>>(materials_);
			return c.set(element, std::move(material));
		}

	private:
		int element_num_;
		std::tuple<utils::PackedArray<M>...> materials_;
	};

} // namespace polyfem::material
