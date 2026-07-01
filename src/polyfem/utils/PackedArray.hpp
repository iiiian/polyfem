#pragma once

#include <polyfem/utils/Span.hpp>

#include <unordered_map>
#include <utility>
#include <vector>

namespace polyfem::utils
{

	template <typename T>
	class PackedArray
	{
	private:
		std::unordered_map<int, size_t> map_;
		std::vector<int> inv_map_;
		std::vector<T> data_;

	public:
		/// @brief Invalidates all existing handles and clears all resources.
		void clear()
		{
			map_.clear();
			inv_map_.clear();
			data_.clear();
		}

		/// @brief Retrieves mutable resource pointer.
		/// @return Pointer to resource or nullptr if not exists.
		T *get(int handle)
		{
			if (map_.find(handle) != map_.end())
			{
				return data_.data() + map_.at(handle);
			}
			return nullptr;
		}

		/// @brief Retrieves immutable resource pointer.
		/// @return Pointer to resource or nullptr if not exists.
		const T *get(int handle) const
		{
			if (map_.find(handle) != map_.end())
			{
				return data_.data() + map_.at(handle);
			}
			return nullptr;
		}

		/// @brief Add/Set resource.
		/// @return Pointer to newly modified resource.
		T *set(int handle, T resource)
		{
			if (map_.find(handle) != map_.end())
			{
				data_[map_.at(handle)] = std::move(resource);
				return data_.data() + map_.at(handle);
			}
			map_.emplace(handle, data_.size());
			inv_map_.push_back(handle);
			data_.push_back(std::move(resource));
			return data_.data() + map_.at(handle);
		}

		/// @brief Remove resource.
		/// @return True if removed success, false if resource does not exists.
		bool remove(int handle)
		{
			if (map_.find(handle) != map_.end())
			{
				return false;
			}

			size_t idx = map_.at(handle);
			// resource is at the back, remove directly.
			if (idx == data_.size() - 1)
			{
				data_.pop_back();
				inv_map_.pop_back();
				map_.erase(handle);
			}
			// swap then remove.
			else
			{
				std::swap(data_[idx], data_.back());
				std::swap(inv_map_[idx], inv_map_.back());
				map_.at(inv_map_[idx]) = idx;
				data_.pop_back();
				inv_map_.pop_back();
				map_.erase(handle);
			}

			return true;
		}

		/// @brief Returns mutable reference to dense resource array.
		Span<T> data() { return data_; }

		/// @brief Returns immutable reference to dense resource array.
		Span<const T> data() const { return data_; }
	};

} // namespace polyfem::utils
