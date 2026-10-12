if(TARGET inflator::inflator)
    return()
endif()

find_package(inflator CONFIG QUIET)
if(TARGET inflator::inflator)
    return()
endif()

include(ExternalProject)
set(POLYFEM_INFLATOR_SOURCE_DIR "" CACHE PATH "Local inflator source checkout instead of downloading the pinned fork")
set(inflator_prefix "${CMAKE_BINARY_DIR}/inflator")
set(inflator_install "${inflator_prefix}/install")
set(inflator_build_type "${CMAKE_BUILD_TYPE}")
if(NOT inflator_build_type)
    set(inflator_build_type Release)
endif()
set(inflator_library "${inflator_install}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}inflator${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(POLYFEM_INFLATOR_SOURCE_DIR)
    set(inflator_source SOURCE_DIR "${POLYFEM_INFLATOR_SOURCE_DIR}")
else()
    set(inflator_source
        GIT_REPOSITORY https://github.com/iiiian/microstructure_inflators.git
        GIT_TAG 14e10832611194de4969c61a617b31dfb673127f)
endif()

ExternalProject_Add(polyfem_inflator_build
    PREFIX "${inflator_prefix}"
    ${inflator_source}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -S <SOURCE_DIR> -B <BINARY_DIR> -G Ninja
        -DCMAKE_BUILD_TYPE=${inflator_build_type}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${inflator_install}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DMICRO_BUILD_BINARIES=OFF
        -DMICRO_BUILD_TOOLS=OFF
        -DMICRO_WITH_TBB=OFF
        -DMICRO_WITH_UBUNTU=OFF
        -DLIBIGL_WITH_OPENGL=OFF
        -DLIBIGL_WITH_OPENGL_GLFW=OFF
        -DLIBIGL_WITH_OPENGL_GLFW_IMGUI=OFF
        -DLIBIGL_WITH_VIEWER=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target inflator
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
    BUILD_BYPRODUCTS "${inflator_library}")

file(MAKE_DIRECTORY "${inflator_install}/include")
add_library(inflator::inflator SHARED IMPORTED GLOBAL)
set_target_properties(inflator::inflator PROPERTIES
    IMPORTED_LOCATION "${inflator_library}"
    INTERFACE_COMPILE_FEATURES cxx_std_17
    INTERFACE_INCLUDE_DIRECTORIES "${inflator_install}/include")
add_dependencies(inflator::inflator polyfem_inflator_build)
