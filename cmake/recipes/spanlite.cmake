# span-lite (https://github.com/nonstd-lite/span-lite)
# License: BSL 1.0

if(TARGET nonstd::span-lite)
    return()
endif()

message(STATUS "Third-party: creating target 'nonstd::span-lite'")

include(CPM)
CPMAddPackage(
    NAME span-lite
    GITHUB_REPOSITORY nonstd-lite/span-lite
    VERSION 0.11.0
)
