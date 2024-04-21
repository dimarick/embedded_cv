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

add_library(mini_server BroadcastingServer.cpp CallbackHandler.cpp CommandServer.cpp
        SocketFactory.cpp)

install(TARGETS mini_server DESTINATION lib)