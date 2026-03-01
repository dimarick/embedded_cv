cmake_minimum_required(VERSION 3.16)
project(embedded_cv)

set(CMAKE_PREFIX_PATH release/amd64/Debug/OpenCV)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

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
    message(STATUS "    libraries: ${OpenCL_LIBRARIES}")
else(OpenCL_FOUND)
    message(">> OpenCL not found. OpenCL is required for GPGPU/FPGA/Accelerator use.")
endif(OpenCL_FOUND)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};release/${ARCH}/${CMAKE_BUILD_TYPE}/OpenCV)

include_directories("release/${ARCH}/Debug/OpenCV/include/opencv5")
include_directories("release/${ARCH}/Debug/OpenCV/include/opencv5/opencv2")

# only for ide analizer
add_definitions(-include /home/dima/embedded_cv/release/${ARCH}/Debug/OpenCV/include/opencv5/opencv2/cvconfig.h)
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

set(CMAKE_CXX_STANDARD 23)
list(APPEND CMAKE_FIND_ROOT_PATH "/")
find_package(OpenCV 5.0.0 REQUIRED)

set(OPT_FLAGS "-O0 -g -ffast-math -no-pie -ftree-vectorize -pthread")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

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
        GIT_TAG        v1.0.4
)
FetchContent_MakeAvailable(cpptrace)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.13.0
)

FetchContent_MakeAvailable(Catch2)


add_executable(tests
        src/calibrator/Calibrator.cpp
        src/calibrator/CalibrationStrategy.cpp
        src/calibrator/CalibrateFrameCollector.cpp
        src/calibrator/GridPreferredSizeProvider.cpp
        tests/calibrator/CalibrateFrameCollectorTest.cpp
        tests/calibrator/StatStreamingTest.cpp
        tests/calibrator/StatStreamingTest.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain PUBLIC OpenCL::OpenCL cpptrace::cpptrace Eigen3::Eigen ${OpenCV_LIBS})

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(./mini_server)
add_subdirectory(./mini_server)
#add_subdirectory(./hw_ctl)
add_subdirectory(./ws_ctl)

add_executable(calib
        src/calibrator/calib.cpp
        src/calibrator/CalibrateFrameCollector.cpp
        src/calibrator/CalibrateMapper.cpp
        src/calibrator/Calibrator.cpp
        src/calibrator/Calibrator.h
        MatStorage.cpp
        MatStorage.h
        src/calibrator/GridPreferredSizeProvider.cpp
        src/calibrator/CalibrationStrategy.cpp
        src/calibrator/CalibrationStrategy.h
        src/calibrator/SingleCameraThread.cpp
        src/calibrator/StatStreaming.h
        src/calibrator/MultiCameraCalibration.cpp
        src/calibrator/MultiCameraCalibration.h
)

target_link_libraries(calib PUBLIC OpenCL::OpenCL cpptrace::cpptrace Eigen3::Eigen ${OpenCV_LIBS})

install(TARGETS calib DESTINATION bin)

add_executable(cv
        cv.cpp
        CvCommandHandler.cpp
        ImageProcessor.cpp
        DisparityEvaluator.cpp
        DisparityEvaluator.h
)

target_link_libraries(cv PUBLIC OpenCL::OpenCL mini_server cpptrace::cpptrace Eigen3::Eigen ${OpenCV_LIBS})

add_executable(benchmark
        benchmark/benchmark.cpp
        benchmark/KitDisparityBenchmark.cpp
        DisparityEvaluator.cpp
        DisparityEvaluator.h
        src/calibrator/CalibrateFrameCollector.cpp
        src/calibrator/CalibrateFrameCollector.h
        src/calibrator/GridPreferredSizeProvider.cpp
        src/calibrator/GridPreferredSizeProvider.h
        src/calibrator/SingleCameraThread.cpp
        src/calibrator/SingleCameraThread.h
)

target_link_libraries(benchmark PUBLIC OpenCL::OpenCL cpptrace::cpptrace ${OpenCV_LIBS})

install(TARGETS cv DESTINATION bin)
