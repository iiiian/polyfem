# IPC Toolkit (https://github.com/ipc-sim/ipc-toolkit)
# License: MIT

if(TARGET ipc::toolkit)
    return()
endif()

message(STATUS "Third-party: creating target 'ipc::toolkit'")

if (POLYFEM_WITH_CUDA)
  set(IPC_TOOLKIT_WITH_CUDA ON)
endif()

include(CPM)
CPMAddPackage("gh:ipc-sim/ipc-toolkit#9a704eae1a32297fa6b8dee4e92d047c8ca4ebeb")
