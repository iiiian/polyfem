# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)
CPMAddPackage(
    NAME polysolve
    GIT_REPOSITORY "file://${CMAKE_CURRENT_SOURCE_DIR}/polysolve"
    GIT_TAG inexact-newton
)
