#pragma once

#include <deque>
#include <utility>
#include <vector>
#include <limits>

#include <polyfem/utils/span.hpp>

namespace polyfem::ecs
{

	/// @brief A dynamic array with two level index mappings.
	///
	/// Use two layer mapping:
	/// handle -> slots -> dense storage array.
	template <typename T>
	class PackedArray
	{
	private:
		// Indirection table mapping handle indices to data array indices.
		std::vector<int> slots_;
		// Queue of available slot indices for new allocations.
		std::deque<int> free_slots_;
		// Dense array storing actual data.
		std::vector<T> data_;
		// Reverse mapping from data array index to slot index.
		std::vector<int> slot_of_data_;

	public:
		PackedArray() = default;

		/// @brief Get ptr from handle. Return nullptr on invalid handle.
		T *get(int handle)
		{
			if (handle < 0 || handle >= slots_.size() || slots_[handle] < 0)
			{
				return nullptr;
			}
			return data_.data() + slots_[handle];
		}

		/// @brief Get ptr from handle. Return nullptr on invalid handle.
		const T *get(int handle) const
		{
			if (handle < 0 || handle >= slots_.size() || slots_[handle] < 0)
			{
				return nullptr;
			}
			return data_.data() + slots_[handle];
		}

		/// @brief Adds a new resource and returns its handle. Return -1 if out of capacity.
		int add(const T &component)
		{
			if (free_slots_.empty() && slots_.size() >= std::numeric_limits<int>::max())
			{
				return -1; // At maximum capacity
			}

			int slot_idx;
			if (free_slots_.empty())
			{
				slot_idx = slots_.size();
				slots_.push_back(-1);
			}
			else
			{
				slot_idx = free_slots_.front();
				free_slots_.pop_front();
			}

			slots_[slot_idx] = data_.size();
			data_.push_back(component);
			slot_of_data_.push_back(slot_idx);
			return slot_idx;
		}

		/// @brief Adds a new resource and returns its handle. Return -1 if out of capacity.
		int add(T &&component)
		{
			if (free_slots_.empty() && slots_.size() >= std::numeric_limits<int>::max())
			{
				return -1; // At maximum capacity
			}

			int slot_idx;
			if (free_slots_.empty())
			{
				slot_idx = slots_.size();
				slots_.push_back(-1);
			}
			else
			{
				slot_idx = free_slots_.front();
				free_slots_.pop_front();
			}

			slots_[slot_idx] = data_.size();
			data_.push_back(std::move(component));
			slot_of_data_.push_back(slot_idx);
			return slot_idx;
		}

		/// @brief Removes resource and invalidates its handle.
		/// @return true if resource is removed, false if handle is invalid.
		bool remove(int handle)
		{
			if (handle < 0 || handle >= slots_.size() || slots_[handle] < 0)
			{
				return false;
			}

			int dead_slot = slots_[handle];
			// Swap-and-pop: move last element to fill gap, maintaining dense layout
			if (dead_slot != data_.size() - 1)
			{
				data_[dead_slot] = std::move_if_noexcept(data_.back());
				slots_[slot_of_data_.back()] = dead_slot;
				slot_of_data_[dead_slot] = slot_of_data_.back();
			}
			data_.pop_back();
			slot_of_data_.pop_back();

			// Invalidate slot.
			slots_[handle] = -1;
			free_slots_.push_front(handle);

			return true;
		}

		/// @brief Returns dense resource array.
		tcb::span<T> data() { return data_; }

		/// @brief Returns dense resource array.
		tcb::span<const T> data() const { return data_; }
	};

} // namespace polyfem::ecs
