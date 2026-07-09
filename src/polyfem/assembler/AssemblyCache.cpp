#include <polyfem/utils/Range.hpp>
#include <polyfem/assembler/AssemblyCache.hpp>

#include <cassert>
#include <Eigen/Core>

#ifdef POLYFEM_WITH_CUDA
#include <cuda/buffer>
#include <cuda/algorithm>
#endif

namespace polyfem::assembler
{

	void AssemblyTempStorage::resize(int dim, int basis_num, int geom_basis_num, int quad_num)
	{

		int n = basis_num * quad_num;
		int gn = geom_basis_num * quad_num;

		basis_values.resize(n);
		basis_grad_x.resize(n);
		basis_grad_phy_x.resize(n);
		gbasis_values.resize(gn);
		gbasis_grad_x.resize(gn);
		physical_x.resize(quad_num);
		det_J.resize(quad_num);
		J_inverse_transpose.resize(quad_num * dim * dim);
		weighted_measure.resize(quad_num);

		if (dim > 1)
		{
			basis_grad_y.resize(n);
			basis_grad_phy_y.resize(n);
			gbasis_grad_y.resize(gn);
			physical_y.resize(quad_num);
		}

		if (dim > 2)
		{
			basis_grad_z.resize(n);
			basis_grad_phy_z.resize(n);
			gbasis_grad_z.resize(gn);
			physical_z.resize(quad_num);
		}
	}

	AssemblyCacheDesc AssemblyCache::insert(bool is_mass, const AssemblyTempStorage &temp)
	{
#ifdef POLYFEM_WITH_CUDA
		need_host_device_sync_ = true;
#endif

		auto append_and_set_range = [](Span<const double> src, std::vector<double> &dst, Range &range) {
			range.offset = dst.size();
			range.num = src.size();
			dst.insert(dst.end(), src.begin(), src.end());
		};

		AssemblyCacheDesc new_desc;
		new_desc.is_empty = false;
		new_desc.is_mass = is_mass;

		append_and_set_range(temp.basis_values, basis_values_, new_desc.basis_val_range);
		append_and_set_range(temp.basis_grad_x, basis_grad_x_, new_desc.basis_grad_x_range);
		append_and_set_range(temp.basis_grad_y, basis_grad_y_, new_desc.basis_grad_y_range);
		append_and_set_range(temp.basis_grad_z, basis_grad_z_, new_desc.basis_grad_z_range);
		append_and_set_range(temp.basis_grad_phy_x, basis_grad_phy_x_, new_desc.basis_grad_phy_x_range);
		append_and_set_range(temp.basis_grad_phy_y, basis_grad_phy_y_, new_desc.basis_grad_phy_y_range);
		append_and_set_range(temp.basis_grad_phy_z, basis_grad_phy_z_, new_desc.basis_grad_phy_z_range);
		append_and_set_range(temp.physical_x, physical_x_, new_desc.physical_x_range);
		append_and_set_range(temp.physical_y, physical_y_, new_desc.physical_y_range);
		append_and_set_range(temp.physical_z, physical_z_, new_desc.physical_z_range);
		append_and_set_range(temp.det_J, det_J_, new_desc.det_J_range);
		append_and_set_range(temp.J_inverse_transpose, J_inverse_transpose_, new_desc.J_inverse_transpose_range);
		append_and_set_range(temp.weighted_measure, weighted_measure_, new_desc.weighted_measure_range);

		return new_desc;
	}

	void AssemblyCache::clear()
	{
		desc_.clear();
		basis_values_.clear();
		basis_grad_x_.clear();
		basis_grad_y_.clear();
		basis_grad_z_.clear();
		basis_grad_phy_x_.clear();
		basis_grad_phy_y_.clear();
		basis_grad_phy_z_.clear();
		physical_x_.clear();
		physical_y_.clear();
		physical_z_.clear();
		det_J_.clear();
		J_inverse_transpose_.clear();
		weighted_measure_.clear();

#ifdef POLYFEM_WITH_CUDA
		clear_device_storage();
#endif
	}

	int AssemblyCache::append(bool is_mass, const AssemblyTempStorage &temp)
	{
		AssemblyCacheDesc new_desc = insert(is_mass, temp);
		desc_.push_back(new_desc);
		return desc_.size() - 1;
	}

	void AssemblyCache::update(int element_id, bool is_mass, const AssemblyTempStorage &temp)
	{
		assert(element_id >= 0 && element_id < desc_.size());

		AssemblyCacheDesc new_desc = insert(is_mass, temp);
		desc_[element_id] = new_desc;
	}

	AssemblyCacheView AssemblyCache::view() const
	{
		return AssemblyCacheView{
			desc_,
			basis_values_,
			basis_grad_x_,
			basis_grad_y_,
			basis_grad_z_,
			basis_grad_phy_x_,
			basis_grad_phy_y_,
			basis_grad_phy_z_,
			physical_x_,
			physical_y_,
			physical_z_,
			det_J_,
			J_inverse_transpose_,
			weighted_measure_};
	}

#ifdef POLYFEM_WITH_CUDA
	AssemblyCacheView AssemblyCache::device_view(ExecutionPolicy policy)
	{
		auto &p = policy;
		if (need_host_device_sync_)
		{
			d_desc_ = cuda::make_buffer<AssemblyCacheDesc>(*p.stream, *p.mr, desc_.size(), cuda::no_init);
			d_basis_values_ = cuda::make_buffer<double>(*p.stream, *p.mr, basis_values_.size(), cuda::no_init);
			d_basis_grad_x_ = cuda::make_buffer<double>(*p.stream, *p.mr, basis_grad_x_.size(), cuda::no_init);
			d_basis_grad_y_ = cuda::make_buffer<double>(*p.stream, *p.mr, basis_grad_y_.size(), cuda::no_init);
			d_basis_grad_z_ = cuda::make_buffer<double>(*p.stream, *p.mr, basis_grad_z_.size(), cuda::no_init);
			d_basis_grad_phy_x_ = cuda::make_buffer<double>(*p.stream, *p.mr, basis_grad_phy_x_.size(), cuda::no_init);
			d_basis_grad_phy_y_ = cuda::make_buffer<double>(*p.stream, *p.mr, basis_grad_phy_y_.size(), cuda::no_init);
			d_basis_grad_phy_z_ = cuda::make_buffer<double>(*p.stream, *p.mr, basis_grad_phy_z_.size(), cuda::no_init);
			d_physical_x_ = cuda::make_buffer<double>(*p.stream, *p.mr, physical_x_.size(), cuda::no_init);
			d_physical_y_ = cuda::make_buffer<double>(*p.stream, *p.mr, physical_y_.size(), cuda::no_init);
			d_physical_z_ = cuda::make_buffer<double>(*p.stream, *p.mr, physical_z_.size(), cuda::no_init);
			d_det_J_ = cuda::make_buffer<double>(*p.stream, *p.mr, det_J_.size(), cuda::no_init);
			d_J_inverse_transpose_ = cuda::make_buffer<double>(*p.stream, *p.mr, J_inverse_transpose_.size(), cuda::no_init);
			d_weighted_measure_ = cuda::make_buffer<double>(*p.stream, *p.mr, weighted_measure_.size(), cuda::no_init);

			cuda::copy_bytes(*p.stream, desc_, *d_desc_);
			cuda::copy_bytes(*p.stream, basis_values_, *d_basis_values_);
			cuda::copy_bytes(*p.stream, basis_grad_x_, *d_basis_grad_x_);
			cuda::copy_bytes(*p.stream, basis_grad_y_, *d_basis_grad_y_);
			cuda::copy_bytes(*p.stream, basis_grad_z_, *d_basis_grad_z_);
			cuda::copy_bytes(*p.stream, basis_grad_phy_x_, *d_basis_grad_phy_x_);
			cuda::copy_bytes(*p.stream, basis_grad_phy_y_, *d_basis_grad_phy_y_);
			cuda::copy_bytes(*p.stream, basis_grad_phy_z_, *d_basis_grad_phy_z_);
			cuda::copy_bytes(*p.stream, physical_x_, *d_physical_x_);
			cuda::copy_bytes(*p.stream, physical_y_, *d_physical_y_);
			cuda::copy_bytes(*p.stream, physical_z_, *d_physical_z_);
			cuda::copy_bytes(*p.stream, det_J_, *d_det_J_);
			cuda::copy_bytes(*p.stream, J_inverse_transpose_, *d_J_inverse_transpose_);
			cuda::copy_bytes(*p.stream, weighted_measure_, *d_weighted_measure_);

			need_host_device_sync_ = false;

			p.stream->sync();
		}
		return AssemblyCacheView{
			*d_desc_,
			*d_basis_values_,
			*d_basis_grad_x_,
			*d_basis_grad_y_,
			*d_basis_grad_z_,
			*d_basis_grad_phy_x_,
			*d_basis_grad_phy_y_,
			*d_basis_grad_phy_z_,
			*d_physical_x_,
			*d_physical_y_,
			*d_physical_z_,
			*d_det_J_,
			*d_J_inverse_transpose_,
			*d_weighted_measure_};
	}

	void AssemblyCache::clear_device_storage()
	{
		need_host_device_sync_ = true;
		d_desc_ = {};
		d_basis_values_ = {};
		d_basis_grad_x_ = {};
		d_basis_grad_y_ = {};
		d_basis_grad_z_ = {};
		d_basis_grad_phy_x_ = {};
		d_basis_grad_phy_y_ = {};
		d_basis_grad_phy_z_ = {};
		d_physical_x_ = {};
		d_physical_y_ = {};
		d_physical_z_ = {};
		d_det_J_ = {};
		d_J_inverse_transpose_ = {};
		d_weighted_measure_ = {};
	}
#endif

} // namespace polyfem::assembler
