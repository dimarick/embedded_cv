cmake_minimum_required(VERSION 3.16)
project(embedded_cv)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_FIND_ROOT_PATH "/")
find_package(OpenCV 4.9.0 REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(embedded_cv main.cpp)

target_link_libraries(embedded_cv PRIVATE ${OpenCV_LIBS})

install(TARGETS embedded_cv DESTINATION bin)