# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

if (POLYFEM_WITH_CUDA)
  set(POLYSOLVE_WITH_CUDA ON)
endif()

include(CPM)
CPMAddPackage("gh:iiiian/polysolve#mas-only")
