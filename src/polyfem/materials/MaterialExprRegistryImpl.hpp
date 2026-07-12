#pragma once

#include <polyfem/utils/Span.hpp>

#include <tuple>
#include <type_traits>
#include <vector>
#include <cassert>

#ifdef POLYFEM_WITH_CUDA
#include <polyfem/utils/ExecutionPolicy.hpp>
#include <polyfem/utils/CUDAUtils.hpp>
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::material
{

	template <typename M>
	struct MaterialStoreView
	{
		/// Array of Material expressions. Might be empty.
		Span<const M> expr;
		/// Per element material expression id. Empty if expr is empty.
		/// id == -1 is a special value indicating no material expr.
		Span<const int> expr_ids;
	};

	template <typename M>
	class MaterialStore
	{
	private:
		std::vector<M> expr_;
		std::vector<int> expr_ids_;

#ifdef POLYFEM_WITH_CUDA
		mutable bool need_host_device_sync_ = true;
		mutable DeviceBuf<M> d_expr_;
		mutable DeviceBuf<int> d_expr_ids;
#endif

	public:
		void set(int element_num, Span<const int> target_elements, M material)
		{
			assert(element_num >= 0);

			expr_.push_back(std::move(material));
			int material_id = expr_.size() - 1;

			if (expr_ids_.empty())
			{
				expr_ids_.resize(element_num, -1);
			}

			for (int t : target_elements)
			{
				assert(t >= 0 && t < element_num);
				expr_ids_[t] = material_id;
			}

#ifdef POLYFEM_WITH_CUDA
			need_host_device_sync_ = true;
#endif
		};

		MaterialStoreView<M> view() const
		{
			return MaterialStoreView<M>{expr_, expr_ids_};
		}

#ifdef POLYFEM_WITH_CUDA
		/// Return view on device memory. Lazily sync data.
		MaterialStoreView<M> device_view(ExecutionPolicy policy) const
		{
			auto &p = policy;
			if (need_host_device_sync_)
			{
				d_expr_ = cuda::make_buffer<M>(*p.stream, *p.mr, expr_.size(), cuda::no_init);
				d_expr_ids = cuda::make_buffer<int>(*p.stream, *p.mr, expr_ids_.size(), cuda::no_init);
				cuda::copy_bytes(*p.stream, expr_, *d_expr_);
				cuda::copy_bytes(*p.stream, expr_ids_, *d_expr_ids);

				need_host_device_sync_ = false;
				p.stream->sync();
			}

			return MaterialStore<M>{*d_expr_, *d_expr_ids};
		}

		/// Release device storage.
		void clear_device_storage()
		{
			need_host_device_sync_ = true;
			d_expr_ = {};
			d_expr_ids = {};
		}
#endif
	};

	template <typename... M>
	class MaterialExprRegistryImpl
	{
	public:
		MaterialExprRegistryImpl(int element_num) : element_num_(element_num) {};

		// -----------------------------------------------------
		// element APIs
		// -----------------------------------------------------

		int element_num() const { return element_num_; };

		// -----------------------------------------------------
		// material APIs
		// -----------------------------------------------------

		/// @brief Return true if element has material expr T.
		template <typename T>
		bool has_material(int element) const
		{
			assert(element >= 0 && element < element_num_);
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			auto v = s.view();
			return !(v.expr_ids.empty() || v.expr_ids_[element] == -1);
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		template <typename T>
		T *get(int element)
		{
			assert(element >= 0 && element < element_num_);
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			if (!has_material<T>(element))
				return nullptr;

			auto &s = std::get<MaterialStore<T>>(materials_);
			auto v = s.view();
			return v.expr_.data() + v.expr_ids[element];
		}

		/// @brief Return ptr to element material expr T. nullptr if not exists.
		template <typename T>
		const T *get(int element) const
		{
			assert(element >= 0 && element < element_num_);
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material expr type, double check T appears as "
						  "template argument in registry declaration.");

			if (!has_material<T>(element))
				return nullptr;

			auto &s = std::get<MaterialStore<T>>(materials_);
			auto v = s.view();
			return v.expr_.data() + v.expr_ids[element];
		}

		/// @brief Return all materials of type T.
		template <typename T>
		MaterialStoreView<T> get_all_materials()
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			return s.view();
		}

#ifdef POLYFEM_WITH_CUDA
		/// @brief Return all materials of type T.
		template <typename T>
		MaterialStoreView<T> get_all_materials_device(ExecutionPolicy p)
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			return s.device_view(p);
		}
#endif

		/// @brief Set/Replace material expr T of element.
		template <typename T>
		void set(Span<const int> elements, T material)
		{
			static_assert((std::is_same_v<T, M> || ...),
						  "T is not a material type, double check T appears as "
						  "template argument in registry declaration.");

			auto &s = std::get<MaterialStore<T>>(materials_);
			s.set(element_num_, elements, std::move(material));
		}

	private:
		int element_num_;
		std::tuple<MaterialStore<M>...> materials_;
	};

} // namespace polyfem::material
