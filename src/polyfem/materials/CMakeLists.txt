set(SOURCES
  MaterialExprRegistryImpl.hpp
  MaterialExprRegistry.hpp
  ElasticityTensor.hpp
  ElasticityTensor.cpp
  FiberDirection.hpp
  FiberDirection.cpp
  LameParameter.hpp
  LameParameter.cpp
  Materials.hpp
  Materials.cpp
  BuildMaterialExprFromJson.hpp
  BuildMaterialExprFromJson.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfem PRIVATE ${SOURCES})
