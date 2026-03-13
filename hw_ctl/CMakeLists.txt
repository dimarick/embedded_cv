cmake_minimum_required(VERSION 3.16)
project(hw_ctl)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG        v0.5.2
)
FetchContent_MakeAvailable(cpptrace)

include_directories(../src/mini_server)

add_executable(hw_ctl main.cpp)
target_link_libraries(hw_ctl PRIVATE cpptrace::cpptrace mini_server)

install(TARGETS hw_ctl DESTINATION bin)
