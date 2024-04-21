cmake_minimum_required(VERSION 3.16)
project(ws_ctl)

set(CMAKE_CXX_STANDARD 17)

set(BUILD_SHARED_LIBS OFF)
set(DEFLATE_SUPPORT OFF)
set(UNITTESTS OFF)
set(SEASOCKS_EXAMPLE_APP OFF)

add_subdirectory(3rdparty/seasocks EXCLUDE_FROM_ALL)
add_subdirectory(web)

include_directories(3rdparty/seasocks/src/main/c)

include(FetchContent)
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG        v0.5.2
)
FetchContent_MakeAvailable(cpptrace)

add_executable(ws_ctl main.cpp SocketProxy.cpp)
target_link_libraries(ws_ctl PRIVATE cpptrace::cpptrace seasocks embedded2)

install(TARGETS ws_ctl DESTINATION bin)
