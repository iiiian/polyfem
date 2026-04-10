#pragma once

#include <vector>
#include <tuple>
#include <cassert>
#include <type_traits>
#include <utility>

#include <polyfem/utils/span.hpp>
#include <polyfem/ecs/PackedArray.hpp>

namespace polyfem::ecs
{
	/// @brief Poor man's ECS registry.
	///
	/// This is a simplified version of ECS system, you need to specify a fixed number of entity
	/// at registry creation. Support up to 2^31 entities.
	template <typename... C>
	class Registry
	{
	public:
		Registry() = default;

		Registry(int entity_num)
		{
			assert(entity_num >= 0);

			entity_num_ = entity_num;

			auto allocate_handles = [entity_num](auto &...comp) {
				(comp.handles.resize(entity_num, -1), ...);
			};
			std::apply(allocate_handles, components_);
		}

		~Registry() = default;

		// Copying registry is super expensive. This should never happens.

		Registry(const Registry &) = delete;
		Registry(Registry &&other) noexcept = default;

		Registry &operator=(const Registry &) = delete;
		Registry &operator=(Registry &&other) noexcept = default;

		// -----------------------------------------------------
		// Query APIs
		// -----------------------------------------------------

		/// @brief Return true if entity has component T.
		template <typename T>
		bool has_component(int entity) const
		{
			assert(entity >= 0 && entity < entity_num_);
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			auto &c = std::get<ComponentStorage<T>>(components_);
			return (c.handles[entity] >= 0);
		}

		/// @brief Gather all entities with compoents T...
		/// @tparam T All required components.
		/// @return Vector of entity id.
		template <typename... T>
		std::vector<int> get_entity_with_components() const
		{
			std::vector<int> result;
			for (int i = 0; i < entity_num_; ++i)
			{
				if ((has_component<T>(i) && ...))
				{
					result.push_back(i);
				}
			}
			return result;
		}

		/// @brief Return ptr to component T of entity. nullptr if not exists.
		/// @warning ptr might be invalid after any add/remove operation on component T.
		template <typename T>
		T *get(int entity)
		{
			assert(entity >= 0 && entity < entity_num_);
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			auto &c = std::get<ComponentStorage<T>>(components_);
			return c.packed_data.get(c.handles[entity]);
		}

		/// @brief Return ptr to component T of entity. nullptr if not exists.
		/// @warning ptr might be invalid after any add/remove operation on component T.
		template <typename T>
		const T *get(int entity) const
		{
			assert(entity >= 0 && entity < entity_num_);
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			auto &c = std::get<ComponentStorage<T>>(components_);
			return c.packed_data.get(c.handles[entity]);
		}

		/// @brief Gather ptr to components from entities.
		/// @tparam T Component type.
		/// @param entities Entities.
		/// @return Vector of ptr to components. nullptr if component is missing.
		/// @warning Ptr might be invalidate after any add/remove operation of T.
		template <typename T>
		std::vector<T *> batch_get(tcb::span<const int> entities)
		{
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			std::vector<T *> result(entities.size());
			for (auto i : entities)
			{
				assert(i >= 0 && i < entity_num_);
				result[i] = get<T>(i);
			}
			return result;
		}

		/// @brief Gather ptr to components from entities.
		/// @tparam T Component type.
		/// @param entities Entities.
		/// @return Vector of ptr to components. nullptr if component is missing.
		/// @warning Ptr might be invalidate after any add/remove operation of T.
		template <typename T>
		std::vector<const T *> batch_get(tcb::span<const int> entities) const
		{
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			std::vector<const T *> result(entities.size());
			for (auto i : entities)
			{
				assert(i >= 0 && i < entity_num_);
				result[i] = get<T>(i);
			}
			return result;
		}

		/// @brief Return all components of type T.
		/// @warning span might be invalid after any add/remove operation on component T.
		template <typename T>
		tcb::span<T> get_all_components()
		{
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			auto &c = std::get<ComponentStorage<T>>(components_);
			return c.packed_data.data();
		}

		/// @brief Return all components of type T.
		/// @warning span might be invalid after any add/remove operation on component T.
		template <typename T>
		tcb::span<const T> get_all_components() const
		{
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			auto &c = std::get<ComponentStorage<T>>(components_);
			return c.packed_data.data();
		}

		// -----------------------------------------------------
		// Modification APIs
		// -----------------------------------------------------

		/// @brief Remove component T from entity and invalidate handle.
		/// No-op if entity does not have such component.
		///
		/// @warning MODIFICATION API ARE NOT THREAD SAFE!
		template <typename T>
		void remove(int entity)
		{
			assert(entity >= 0 && entity < entity_num_);
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			auto &c = std::get<ComponentStorage<T>>(components_);
			c.packed_data.remove(c.handles[entity]);
			c.handles[entity] = -1;
		}

		/// @brief Remove all components from entity and invalidate handle.
		/// @warning MODIFICATION API ARE NOT THREAD SAFE!
		void remove_all(int entity)
		{
			assert(entity >= 0 && entity < entity_num_);

			auto remove_comp = [entity](auto &c) {
				c.packed_data.remove(c.handles[entity]);
				c.handles[entity] = -1;
			};
			// clang-format off
			std::apply([&](auto &...comp) {
				(remove_comp(comp), ...);
			}, components_);
			// clang-format on
		}

		/// @brief Set/replace component T of entity.
		/// @return Pointer to newly created component.
		/// @warning MODIFICATION API ARE NOT THREAD SAFE!
		template <typename T>
		T *set(int entity, T component)
		{
			assert(entity >= 0 && entity < entity_num_);
			static_assert((std::is_same_v<T, C> || ...),
						  "T is not a component type, double check T appears as template argument in registry declaration.");

			auto &c = std::get<ComponentStorage<T>>(components_);
			int &h = c.handles[entity];

			// Component does not exists, create a new one.
			if (h < 0)
			{
				h = c.packed_data.add(std::move(component));

				// h might not be valid if packed array is at max capacity.
				// But since the max entity num is also bounded by int, this should never happens.
				assert(h >= 0);

				return c.packed_data.get(h);
			}

			// Replace existing components.
			T *old = c.packed_data.get(h);
			assert(old != nullptr);
			*old = std::move(component);
			return old;
		}

		// -----------------------------------------------------
		// Misc APIs
		// -----------------------------------------------------

		int entity_num() const { return entity_num_; };

	private:
		template <typename T>
		struct ComponentStorage
		{
			PackedArray<T> packed_data;
			std::vector<int> handles;
		};

		int entity_num_ = 0;
		std::tuple<ComponentStorage<C>...> components_;
	};

} // namespace polyfem::ecs
