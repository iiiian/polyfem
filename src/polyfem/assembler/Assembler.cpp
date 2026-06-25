#include "Assembler.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/assembler/ScatterLocalAssembly.hpp>

#include <igl/Timer.h>

#include <ipc/utils/eigen_ext.hpp>

#include <algorithm>
#include <array>

namespace polyfem::assembler
{
	using namespace basis;
	using namespace quadrature;
	using namespace utils;

	namespace
	{
		class LocalThreadMatStorage
		{
		public:
			std::unique_ptr<MatrixCache> cache = nullptr;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadMatStorage() = delete;

			LocalThreadMatStorage(const int buffer_size, const int rows, const int cols)
			{
				init(buffer_size, rows, cols);
			}

			LocalThreadMatStorage(const int buffer_size, const MatrixCache &c)
			{
				init(buffer_size, c);
			}

			LocalThreadMatStorage(const LocalThreadMatStorage &other)
				: cache(other.cache->copy()), vals(other.vals), da(other.da)
			{
			}

			LocalThreadMatStorage &operator=(const LocalThreadMatStorage &other)
			{
				assert(other.cache != nullptr);
				cache = other.cache->copy();
				vals = other.vals;
				da = other.da;
				return *this;
			}

			void init(const int buffer_size, const int rows, const int cols)
			{
				// assert(rows == cols);
				// cache = std::make_unique<DenseMatrixCache>();
				cache = std::make_unique<SparseMatrixCache>();
				cache->reserve(buffer_size);
				cache->init(rows, cols);
			}

			void init(const int buffer_size, const MatrixCache &c)
			{
				if (cache == nullptr)
					cache = c.copy();
				cache->reserve(buffer_size);
				cache->init(c);
			}
		};

		class LocalThreadVecStorage
		{
		public:
			Eigen::MatrixXd vec;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadVecStorage(const int size)
			{
				vec.resize(size, 1);
				vec.setZero();
			}
		};

		class LocalThreadScalarStorage
		{
		public:
			double val;
			ElementAssemblyValues vals;
			QuadratureVector da;

			LocalThreadScalarStorage()
			{
				val = 0;
			}
		};

		struct LinearAssemblerThreadLocalStorage
		{
			AssemblyCache temp_cache;
			GeomMappingScratch temp_geom_scratch;
			std::vector<Eigen::Triplet<double, int>> missing_triplets;
		};

		struct NLAssemblerThreadLocalStorage
		{
			AssemblyCache temp_cache;
			GeomMappingScratch temp_geom_scratch;
			double energy = 0.0;
			Eigen::MatrixXd gradient;
			std::vector<double> local_gradient;
			std::vector<double> local_hessian;
			std::vector<Eigen::Triplet<double, int>> missing_triplets;
		};
	} // namespace

	void Assembler::set_materials(const std::vector<int> &body_ids, const json &body_params, const Units &units, const std::string &root_path)
	{
		if (!body_params.is_array())
		{
			this->add_multimaterial(0, body_params, units, root_path);
			return;
		}

		std::map<int, json> materials;
		for (int i = 0; i < body_params.size(); ++i)
		{
			json mat = body_params[i];
			json id = mat["id"];
			if (id.is_array())
			{
				for (int j = 0; j < id.size(); ++j)
					materials[id[j]] = mat;
			}
			else
			{
				const int mid = id;
				materials[mid] = mat;
			}
		}

		std::set<int> missing;

		std::map<int, int> body_element_count;
		std::vector<int> eid_to_eid_in_body(body_ids.size());
		for (int e = 0; e < body_ids.size(); ++e)
		{
			const int bid = body_ids[e];
			body_element_count.try_emplace(bid, 0);
			eid_to_eid_in_body[e] = body_element_count[bid]++;
		}

		for (int e = 0; e < body_ids.size(); ++e)
		{
			const int bid = body_ids[e];
			const auto it = materials.find(bid);
			if (it == materials.end())
			{
				missing.insert(bid);
				continue;
			}

			const json &tmp = it->second;
			this->add_multimaterial(e, tmp, units, root_path);
		}

		for (int bid : missing)
		{
			logger().warn("Missing material parameters for body {}", bid);
		}
	}

	LinearAssembler::LinearAssembler()
	{
	}

	void LinearAssembler::assemble(
		int global_dof_num,
		const basis::ng::ElementBasesView &solution_bases,
		const basis::ng::ElementBasesView &geom_bases,
		const AssemblyCacheView &assembly_cache,
		double time,
		AssemblyResult &result) const
	{
		assert(size() > 0);
		assert(size() * size() <= 9);
		assert(global_dof_num >= 0);
		assert(assembly_cache.desc.size() == solution_bases.element_desc.size());

		BlockCSRMatrix &bsr = result.bsr;

		try
		{
			auto storage = create_thread_storage(LinearAssemblerThreadLocalStorage{});

			int element_num = solution_bases.element_desc.size();
			igl::Timer timer;
			timer.start();

			maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
				LinearAssemblerThreadLocalStorage &local_storage =
					get_local_thread_storage(storage, thread_id);

				for (int e = start; e < end; ++e)
				{
					LinearElementAssemblyData data;
					auto &element_desc = solution_bases.element_desc[e];
					int dim = element_desc.basis_desc.dim;
					int basis_num = basis::ng::local_basis_num(element_desc.basis_desc);
					auto &cache_desc = assembly_cache.desc[e];
					if (cache_desc.is_empty)
					{
						local_storage.temp_cache.clear();
						compute_assembly_cache_single(
							solution_bases,
							geom_bases,
							e,
							false,
							false,
							true,
							true,
							local_storage.temp_geom_scratch,
							local_storage.temp_cache);
						data = LinearElementAssemblyData{e, 0, 0, 0, dim, basis_num, time, local_storage.temp_cache.view()};
					}
					else
					{
						data = LinearElementAssemblyData{e, e, 0, 0, dim, basis_num, time, assembly_cache};
					}

					auto dof_mapping = solution_bases.dof_mapping;

					std::array<double, 9> element_matrix_storage;
					span<double> element_matrix(element_matrix_storage.data(), size() * size());

					for (int i = 0; i < data.basis_num; ++i)
					{
						for (int j = 0; j <= i; ++j)
						{
							data.row_local_basis_id = i;
							data.col_local_basis_id = j;

							std::fill(element_matrix.begin(), element_matrix.end(), 0.0);
							assemble_element(data, element_matrix);

							int row_mapping_id = element_desc.dof_mapping_range.offset + i;
							int col_mapping_id = element_desc.dof_mapping_range.offset + j;
							scatter_local_assembly_matrix(row_mapping_id, col_mapping_id, dof_mapping, element_matrix.data(), size(), bsr, local_storage.missing_triplets);
							if (j < i)
							{
								std::array<double, 9> transposed_element_matrix_storage;
								span<double> transposed_element_matrix(transposed_element_matrix_storage.data(), size() * size());
								for (int row = 0; row < size(); ++row)
								{
									for (int col = 0; col < size(); ++col)
									{
										transposed_element_matrix[row * size() + col] = element_matrix[col * size() + row];
									}
								}

								scatter_local_assembly_matrix(
									col_mapping_id,
									row_mapping_id,
									dof_mapping,
									transposed_element_matrix.data(),
									size(),
									bsr,
									local_storage.missing_triplets);
							}
						}
					}
				}
			});

			timer.stop();
			logger().trace("done BSR assembly {}s...", timer.getElapsedTime());

			timer.start();
			std::size_t missing_triplet_count = 0;
			for (auto &local_storage : storage)
			{
				missing_triplet_count += local_storage.missing_triplets.size();
			}

			result.triplets.reserve(result.triplets.size() + missing_triplet_count);
			for (auto &local_storage : storage)
			{
				auto &t = local_storage.missing_triplets;
				result.triplets.insert(result.triplets.end(), t.begin(), t.end());
				t = {};
			}
			timer.stop();
			logger().trace("done concatenating BSR miss triplets {}s...", timer.getElapsedTime());
		}
		catch (std::bad_alloc &ba)
		{
			log_and_throw_error("bad alloc {}", ba.what());
		}
	}

	MixedAssembler::MixedAssembler()
	{
	}

	void MixedAssembler::assemble(
		const bool is_volume,
		const int n_psi_basis,
		const int n_phi_basis,
		const std::vector<ElementBases> &psi_bases,
		const std::vector<ElementBases> &phi_bases,
		const std::vector<ElementBases> &gbases,
		const AssemblyValsCache &psi_cache,
		const AssemblyValsCache &phi_cache,
		const double t,
		StiffnessMatrix &stiffness) const
	{
		assert(size() > 0);
		assert(phi_bases.size() == psi_bases.size());

		const int max_triplets_size = int(1e7);
		const int buffer_size = std::min(long(max_triplets_size), long(std::max(n_psi_basis, n_phi_basis)) * std::max(rows(), cols()));
		// logger().debug("buffer_size {}", buffer_size);

		stiffness.resize(n_phi_basis * rows(), n_psi_basis * cols());
		stiffness.setZero();

		auto storage = create_thread_storage(LocalThreadMatStorage(buffer_size, stiffness.rows(), stiffness.cols()));

		const int n_bases = int(phi_bases.size());
		igl::Timer timer;
		timer.start();

		maybe_parallel_for(n_bases, [&](int start, int end, int thread_id) {
			LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);
			ElementAssemblyValues psi_vals, phi_vals;

			for (int e = start; e < end; ++e)
			{
				// psi_vals.compute(e, is_volume, psi_bases[e], gbases[e]);
				// phi_vals.compute(e, is_volume, phi_bases[e], gbases[e]);
				psi_cache.compute(e, is_volume, psi_bases[e], gbases[e], psi_vals);
				phi_cache.compute(e, is_volume, phi_bases[e], gbases[e], phi_vals);

				const Quadrature &quadrature = phi_vals.quadrature;

				assert(MAX_QUAD_POINTS == -1 || quadrature.weights.size() < MAX_QUAD_POINTS);
				local_storage.da = phi_vals.det.array() * quadrature.weights.array();
				const int n_phi_loc_bases = int(phi_vals.basis_values.size());
				const int n_psi_loc_bases = int(psi_vals.basis_values.size());

				for (int i = 0; i < n_psi_loc_bases; ++i)
				{
					const auto &global_i = psi_vals.basis_values[i].global;

					for (int j = 0; j < n_phi_loc_bases; ++j)
					{
						const auto &global_j = phi_vals.basis_values[j].global;

						const auto stiffness_val = assemble(MixedAssemblerData(psi_vals, phi_vals, t, i, j, local_storage.da));
						assert(stiffness_val.size() == rows() * cols());

						// igl::Timer t1; t1.start();
						for (int n = 0; n < rows(); ++n)
						{
							for (int m = 0; m < cols(); ++m)
							{
								const double local_value = stiffness_val(n * cols() + m);

								for (size_t ii = 0; ii < global_i.size(); ++ii)
								{
									const auto gi = global_i[ii].index * cols() + m;
									const auto wi = global_i[ii].val;

									for (size_t jj = 0; jj < global_j.size(); ++jj)
									{
										const auto gj = global_j[jj].index * rows() + n;
										const auto wj = global_j[jj].val;

										local_storage.cache->add_value(e, gj, gi, local_value * wi * wj);

										if (local_storage.cache->entries_size() >= max_triplets_size)
										{
											local_storage.cache->prune();
											logger().debug("cleaning memory...");
										}
									}
								}
							}
						}
					}
				}
			}
		});

		timer.stop();
		logger().trace("done separate assembly {}s...", timer.getElapsedTime());

		timer.start();
		// Serially merge local storages
		for (LocalThreadMatStorage &local_storage : storage)
			stiffness += local_storage.cache->get_matrix(false); // will also prune
		stiffness.makeCompressed();
		timer.stop();
		logger().trace("done merge assembly {}s...", timer.getElapsedTime());

		// stiffness.resize(n_basis*size(), n_basis*size());
		// stiffness.setFromTriplets(entries.begin(), entries.end());
	}

	double NLAssembler::assemble_energy(
		const basis::ng::ElementBasesView &solution_bases,
		const basis::ng::ElementBasesView &geom_bases,
		const AssemblyCacheView &assembly_cache,
		double time,
		double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev) const
	{
		assert(size() > 0);
		assert(assembly_cache.desc.size() == solution_bases.element_desc.size());

		auto storage = create_thread_storage(NLAssemblerThreadLocalStorage{});
		int element_num = int(solution_bases.element_desc.size());

		maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
			NLAssemblerThreadLocalStorage &local_storage = get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				auto &element_desc = solution_bases.element_desc[e];
				int dim = element_desc.basis_desc.dim;
				int basis_num = basis::ng::local_basis_num(element_desc.basis_desc);
				auto &cache_desc = assembly_cache.desc[e];
				if (cache_desc.is_empty)
				{
					local_storage.temp_cache.clear();
					compute_assembly_cache_single(
						solution_bases,
						geom_bases,
						e,
						false,
						false,
						true,
						true,
						local_storage.temp_geom_scratch,
						local_storage.temp_cache);
					const NonLinearElementAssemblyData data(
						e, 0, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, local_storage.temp_cache.view());
					local_storage.energy += compute_energy(data);
				}
				else
				{
					const NonLinearElementAssemblyData data(
						e, e, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, assembly_cache);
					local_storage.energy += compute_energy(data);
				}
			}
		});

		double res = 0;
		for (const NLAssemblerThreadLocalStorage &local_storage : storage)
			res += local_storage.energy;
		return res;
	}

	Eigen::VectorXd NLAssembler::assemble_energy_per_element(
		const basis::ng::ElementBasesView &solution_bases,
		const basis::ng::ElementBasesView &geom_bases,
		const AssemblyCacheView &assembly_cache,
		double time,
		double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev) const
	{
		assert(size() > 0);
		assert(assembly_cache.desc.size() == solution_bases.element_desc.size());

		auto storage = create_thread_storage(NLAssemblerThreadLocalStorage{});
		int element_num = int(solution_bases.element_desc.size());
		Eigen::VectorXd out(element_num);

		maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
			NLAssemblerThreadLocalStorage &local_storage = get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				auto &element_desc = solution_bases.element_desc[e];
				int dim = element_desc.basis_desc.dim;
				int basis_num = basis::ng::local_basis_num(element_desc.basis_desc);
				auto &cache_desc = assembly_cache.desc[e];
				if (cache_desc.is_empty)
				{
					local_storage.temp_cache.clear();
					compute_assembly_cache_single(
						solution_bases,
						geom_bases,
						e,
						false,
						false,
						true,
						true,
						local_storage.temp_geom_scratch,
						local_storage.temp_cache);
					const NonLinearElementAssemblyData data(
						e, 0, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, local_storage.temp_cache.view());
					out[e] = compute_energy(data);
				}
				else
				{
					const NonLinearElementAssemblyData data(
						e, e, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, assembly_cache);
					out[e] = compute_energy(data);
				}
			}
		});

#ifndef NDEBUG
		double assemble_val = assemble_energy(
			solution_bases, geom_bases, assembly_cache, time, dt, displacement, displacement_prev);
		assert(std::abs(assemble_val - out.sum()) < std::max(1e-10 * assemble_val, 1e-10));
#endif

		return out;
	}

	void NLAssembler::assemble_gradient(
		int global_dof_num,
		const basis::ng::ElementBasesView &solution_bases,
		const basis::ng::ElementBasesView &geom_bases,
		const AssemblyCacheView &assembly_cache,
		double time,
		double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev,
		Eigen::MatrixXd &rhs) const
	{
		assert(size() > 0);
		assert(global_dof_num >= 0);
		assert(assembly_cache.desc.size() == solution_bases.element_desc.size());

		rhs.resize(global_dof_num * size(), 1);
		rhs.setZero();

		auto storage = create_thread_storage(NLAssemblerThreadLocalStorage{});

		int element_num = int(solution_bases.element_desc.size());

		maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
			NLAssemblerThreadLocalStorage &local_storage = get_local_thread_storage(storage, thread_id);
			if (local_storage.gradient.size() != rhs.size())
			{
				local_storage.gradient.resize(rhs.size(), 1);
				local_storage.gradient.setZero();
			}

			for (int e = start; e < end; ++e)
			{
				auto &element_desc = solution_bases.element_desc[e];
				int dim = element_desc.basis_desc.dim;
				int basis_num = basis::ng::local_basis_num(element_desc.basis_desc);
				auto &cache_desc = assembly_cache.desc[e];
				int local_gradient_size = basis_num * size();
				local_storage.local_gradient.resize(local_gradient_size);
				std::fill(local_storage.local_gradient.begin(), local_storage.local_gradient.end(), 0.0);
				span<double> local_gradient(local_storage.local_gradient.data(), local_storage.local_gradient.size());
				if (cache_desc.is_empty)
				{
					local_storage.temp_cache.clear();
					compute_assembly_cache_single(
						solution_bases,
						geom_bases,
						e,
						false,
						false,
						true,
						true,
						local_storage.temp_geom_scratch,
						local_storage.temp_cache);
					const NonLinearElementAssemblyData data(
						e, 0, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, local_storage.temp_cache.view());
					assemble_gradient(data, local_gradient);
				}
				else
				{
					const NonLinearElementAssemblyData data(
						e, e, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, assembly_cache);
					assemble_gradient(data, local_gradient);
				}

				scatter_local_assembly_vector(
					element_desc,
					solution_bases.dof_mapping,
					local_gradient.data(),
					basis_num,
					size(),
					span<double>(local_storage.gradient.data(), local_storage.gradient.size()));
			}
		});

		for (const NLAssemblerThreadLocalStorage &local_storage : storage)
			rhs += local_storage.gradient;
	}

	void NLAssembler::assemble_hessian(
		int global_dof_num,
		bool project_to_psd,
		const basis::ng::ElementBasesView &solution_bases,
		const basis::ng::ElementBasesView &geom_bases,
		const AssemblyCacheView &assembly_cache,
		double time,
		double dt,
		const Eigen::MatrixXd &displacement,
		const Eigen::MatrixXd &displacement_prev,
		AssemblyResult &result) const
	{
		assert(size() > 0);
		assert(global_dof_num >= 0);
		assert(assembly_cache.desc.size() == solution_bases.element_desc.size());

		BlockCSRMatrix &bsr = result.bsr;
		auto storage = create_thread_storage(NLAssemblerThreadLocalStorage{});

		int element_num = int(solution_bases.element_desc.size());
		igl::Timer timer;
		timer.start();

		maybe_parallel_for(element_num, [&](int start, int end, int thread_id) {
			NLAssemblerThreadLocalStorage &local_storage = get_local_thread_storage(storage, thread_id);

			for (int e = start; e < end; ++e)
			{
				auto &element_desc = solution_bases.element_desc[e];
				int dim = element_desc.basis_desc.dim;
				int basis_num = basis::ng::local_basis_num(element_desc.basis_desc);
				auto &cache_desc = assembly_cache.desc[e];
				int local_matrix_size = basis_num * size();
				int local_hessian_size = local_matrix_size * local_matrix_size;
				local_storage.local_hessian.resize(local_hessian_size);
				std::fill(local_storage.local_hessian.begin(), local_storage.local_hessian.end(), 0.0);
				span<double> local_hessian(local_storage.local_hessian.data(), local_storage.local_hessian.size());
				if (cache_desc.is_empty)
				{
					local_storage.temp_cache.clear();
					compute_assembly_cache_single(
						solution_bases,
						geom_bases,
						e,
						false,
						false,
						true,
						true,
						local_storage.temp_geom_scratch,
						local_storage.temp_cache);
					const NonLinearElementAssemblyData data(
						e, 0, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, local_storage.temp_cache.view());
					assemble_hessian(data, local_hessian);
				}
				else
				{
					const NonLinearElementAssemblyData data(
						e, e, dim, basis_num, time, dt, displacement, displacement_prev, solution_bases, assembly_cache);
					assemble_hessian(data, local_hessian);
				}

				if (project_to_psd)
				{
					Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> local_hessian_mat(
						local_hessian.data(), local_matrix_size, local_matrix_size);
					local_hessian_mat = ipc::project_to_psd(local_hessian_mat);
				}

				scatter_dense_element_matrix(
					element_desc,
					solution_bases.dof_mapping,
					local_hessian.data(),
					basis_num,
					size(),
					bsr,
					local_storage.missing_triplets);
			}
		});

		timer.stop();
		logger().trace("done BSR NL hessian assembly {}s...", timer.getElapsedTime());

		timer.start();
		std::size_t missing_triplet_count = 0;
		for (auto &local_storage : storage)
		{
			missing_triplet_count += local_storage.missing_triplets.size();
		}

		result.triplets.reserve(result.triplets.size() + missing_triplet_count);
		for (auto &local_storage : storage)
		{
			auto &t = local_storage.missing_triplets;
			result.triplets.insert(result.triplets.end(), t.begin(), t.end());
			t.clear();
		}

		timer.stop();
		logger().trace("done concatenating NL BSR miss triplets {}s...", timer.getElapsedTime());
	}

	void ElasticityAssembler::set_use_robust_jacobian()
	{
#ifdef POLYFEM_WITH_BEZIER
		use_robust_jacobian = true;
#else
		logger().error("Enable Bezier library to use the robust Jacobian check!");
#endif
	}
} // namespace polyfem::assembler
