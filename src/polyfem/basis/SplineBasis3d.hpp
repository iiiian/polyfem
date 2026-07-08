#pragma once

#include <polyfem/mesh/mesh3D/Mesh3D.hpp>
#include <polyfem/assembler/AssemblyEssentials.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/InterfaceData.hpp>

#include <Eigen/Dense>
#include <vector>
#include <map>

namespace polyfem
{
	namespace basis
	{
		class SplineBasis3d
		{
		public:
			static int build_bases(
				const mesh::Mesh3D &mesh,
				const std::string &assembler,
				const int quadrature_order,
				const int mass_quadrature_order,
				assembler::AssemblyEssentials &bases,
				std::vector<mesh::LocalBoundary> &local_boundary,
				std::map<int, InterfaceData> &poly_face_to_data);

			static void fit_nodes(const mesh::Mesh3D &mesh, const int n_bases, assembler::AssemblyEssentials &gbases);
		};
	} // namespace basis
} // namespace polyfem
