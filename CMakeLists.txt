cmake_minimum_required(VERSION 3.16)
project(embedded_cv)

if(NOT DEFINED ARCH)
    set(ARCH "amd64")
    message("Set default arch to ${ARCH}")
else()
    message("arch is ${ARCH}")
endif()

#OpenCL libraries
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
else(OpenCL_FOUND)
    message(">> OpenCL not found. OpenCL is required for GPGPU/FPGA/Accelerator use.")
endif(OpenCL_FOUND)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};release/${ARCH}/${CMAKE_BUILD_TYPE}/OpenCV)

include_directories("release/${ARCH}/Debug/OpenCV/include/opencv5")
include_directories("release/${ARCH}/Debug/OpenCV/include/opencv5/opencv2")

# only for ide analizer
add_definitions(-include /home/dima/embedded_cv/release/${ARCH}/Debug/OpenCV/include/opencv5/opencv2/cvconfig.h)
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
add_definitions(-DBASE_DIR="lib/PRiMEStereoMatch/")

set(CMAKE_CXX_STANDARD 23)
list(APPEND CMAKE_FIND_ROOT_PATH "/")
find_package(OpenCV 5.0.0 REQUIRED)

set(OPT_FLAGS "-O0 -ffast-math -ftree-vectorize -fopt-info-vec-missed=vec.miss.txt -ftree-vectorizer-verbose=5 -mtune=native -mno-avx512f")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( Eigen3 REQUIRED )
include_directories( EIGEN3_INCLUDE_DIR )

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include(FetchContent)
FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG        v1.0.4 # <HASH or TAG>
)
FetchContent_MakeAvailable(cpptrace)

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(./mini_server)
add_subdirectory(./mini_server)
add_subdirectory(./hw_ctl)
add_subdirectory(./ws_ctl)

add_executable(embedded_cv
        main.cpp
        ImageProcessor.cpp
        CvCommandHandler.cpp
        CalibrateMapper.cpp
        Calibrator.cpp
        Calibrator.h
        DisparityEvaluator.cpp
        DisparityEvaluator.h
)

target_link_libraries(embedded_cv PRIVATE cpptrace::cpptrace ${OpenCV_LIBS} ${OpenCL_LIBRARIES} mini_server Eigen3::Eigen)

install(TARGETS embedded_cv DESTINATION bin)
