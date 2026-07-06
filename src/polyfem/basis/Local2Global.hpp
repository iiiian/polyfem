#pragma once

#include <polyfem/utils/Types.hpp>

namespace polyfem
{
	namespace basis
	{
		///
		/// @brief      Represents a virtual node of the FEM mesh as a weighted sum
		///             of real (unknown) nodes. This class stores the id, weights
		///             and positions of the real mesh nodes to use in the weighted
		///             sum.
		///
		class Local2Global
		{
		public:
			int index;  ///< global index of the actual node
			double val; ///< weight

			RowVectorNd node; ///< node position

			Local2Global()
				: index(-1), val(0)
			{
			}

			Local2Global(const int _index, const RowVectorNd &_node, const double _val)
				: index(_index), val(_val), node(_node)
			{
			}
		};
	} // namespace basis
} // namespace polyfem
