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

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_FIND_ROOT_PATH "/")
find_package(OpenCV 5.0.0 REQUIRED)

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
        GIT_TAG        v0.5.2 # <HASH or TAG>
)
FetchContent_MakeAvailable(cpptrace)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( lib/PRiMEStereoMatch/include )

include_directories(./mini_server)
add_subdirectory(./mini_server)
add_subdirectory(./hw_ctl)
add_subdirectory(./ws_ctl)

add_executable(embedded_cv
        main.cpp
        ImageProcessor.cpp
        CvCommandHandler.cpp
        OnlineCalibration.cpp
        lib/PRiMEStereoMatch/src/CVC.cpp
        lib/PRiMEStereoMatch/src/CVC_cl.cpp
        lib/PRiMEStereoMatch/src/CVF.cpp
        lib/PRiMEStereoMatch/src/CVF_cl.cpp
        lib/PRiMEStereoMatch/src/DispEst.cpp
        lib/PRiMEStereoMatch/src/DispSel.cpp
        lib/PRiMEStereoMatch/src/DispSel_cl.cpp
        lib/PRiMEStereoMatch/src/fastguidedfilter.cpp
        lib/PRiMEStereoMatch/src/oclUtil.cpp
        lib/PRiMEStereoMatch/src/PP.cpp
)

target_link_libraries(embedded_cv PRIVATE cpptrace::cpptrace ${OpenCV_LIBS} ${OpenCL_LIBRARIES} mini_server)

install(TARGETS embedded_cv DESTINATION bin)
