// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#include <imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cpptrace/basic.hpp>
#include <stereo.hpp>
#include <core/ocl.hpp>
#include "DisparityEvaluator.h"
namespace ecv {
//    const unsigned char kernelSources[] =
//            {
//#embed "DisparityEvaluator.cl"
//            , 0
//            };

    void DisparityEvaluator::evaluateDisparity(const std::vector<cv::UMat> &frames, cv::Mat &disparity, cv::Mat &variance) {
        for (int i = 1; i < frames.size(); i++) {
            CV_Assert(frames[0].cols == frames[i].cols);
            CV_Assert(frames[0].rows == frames[i].rows);
            CV_Assert(frames[0].channels() == frames[i].channels());
            CV_Assert(frames[0].type() == frames[i].type());
            CV_Assert(frames[0].elemSize() == frames[i].elemSize());
        }


        if (disparity.empty()) {
            disparity = cv::Mat(frames[0].size(), CV_16S);
        } else {
            CV_Assert(disparity.cols == frames[0].cols);
            CV_Assert(disparity.rows == frames[0].rows);
            CV_Assert(disparity.type() == CV_16S);
        }

        try {
            this->evaluateIncrementallyOcl(frames, cv::Mat(), disparity, variance);
        } catch (const cl::Error &e) {
            std::cerr << "OpenCL API exception " << e.err() << " (" << openclErrorString(e.err()) << "), what " << e.what() << std::endl;

            throw e;
        }
    }

    void DisparityEvaluator::evaluateIncrementallyOcl(const std::vector<cv::UMat> &frames, const cv::Mat &roughDisparity, cv::Mat &disparity, cv::Mat &variance) {
        this->lazyInitializeOcl();

        for (int i = 1; i < frames.size(); i++) {
            CV_Assert(frames[0].cols == frames[i].cols);
            CV_Assert(frames[0].rows == frames[i].rows);
            CV_Assert(frames[0].channels() == frames[i].channels());
            CV_Assert(frames[0].type() == frames[i].type());
            CV_Assert(frames[0].elemSize() == frames[i].elemSize());
        }

        std::vector<size_t> maxRanges = this->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

        CV_Assert(frames[0].rows / 2 <= maxRanges[0]);
        CV_Assert(frames[0].cols / 2 <= maxRanges[1]);

        if (disparity.empty()) {
            disparity = cv::Mat(frames[0].size(), CV_16S);
            disparity.setTo(0);
        } else {
            CV_Assert(disparity.cols == frames[0].cols);
            CV_Assert(disparity.rows == frames[0].rows);
            CV_Assert(disparity.type() == CV_16S);
        }

        if (variance.empty()) {
            variance = cv::Mat(frames[0].size(), CV_32F);
            variance.setTo(0);
        } else {
            CV_Assert(variance.cols == frames[0].cols);
            CV_Assert(variance.rows == frames[0].rows);
            CV_Assert(variance.type() == CV_16S);
        }

        cv::Mat rd = roughDisparity;

        if (!rd.empty()) {
            CV_Assert(rd.cols == frames[0].cols / 2);
            CV_Assert(rd.rows == frames[0].rows / 2);
            CV_Assert(rd.type() == CV_16S);
        } else {
            rd = cv::Mat(frames[0].cols / 2, std::max(1, frames[0].rows / 2), CV_16S);
        }
        cl::Buffer gDisparity(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, disparity.dataend - disparity.datastart);
        cl::Buffer gVariance(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, variance.dataend - variance.datastart);

        auto startDisp = std::chrono::high_resolution_clock::now();
//        std::cout << "Starting processing queue " << ((double) (std::chrono::high_resolution_clock::now() - startDisp).count()) / 1e6 << std::endl;
        std::vector<cl::Buffer> gFrames(frames.size());

        std::vector<cv::UMat> processedFrames(frames.size());

        auto clahe = cv::createCLAHE(1, cv::Size(3,3));

        for (int i = 0; i < frames.size(); ++i) {
            cv::cvtColor(frames[i], processedFrames[i], cv::COLOR_RGB2GRAY);
//            cv::equalizeHist(processedFrames[i], processedFrames[i]);
            clahe->apply(processedFrames[i], processedFrames[i]);
            clahe->apply(processedFrames[i], processedFrames[i]);
//            frames[i].copyTo(processedFrames[i]);
        }

        cv::ocl::Queue::getDefault().finish();

        for (int i = 0; i < processedFrames.size(); ++i) {
            gFrames[i] = cl::Buffer((cl_mem)processedFrames[i].handle(cv::ACCESS_READ), true);
        }
        queue.finish();

//        cl::Buffer gRoughDisparity(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, rd.dataend - rd.datastart);
//        queue.enqueueWriteBuffer(gRoughDisparity, CL_FALSE, 0, rd.dataend - rd.datastart, (void *)rd.datastart, nullptr);

        auto channels = (int)processedFrames[0].elemSize();
        auto w = frames[0].cols;
        auto h = frames[0].rows;

//        std::cout << "Buffers enqueued " << ((double) (std::chrono::high_resolution_clock::now() - startDisp).count()) / 1e6 << std::endl;

//        std::cout << "Buffers written " << ((double) (std::chrono::high_resolution_clock::now() - startDisp).count()) / 1e6 << std::endl;

        auto a = 0;
        this->kernel.setArg(a++, gFrames[0]);
        this->kernel.setArg(a++, gFrames[1]);
        this->kernel.setArg(a++, gDisparity);
        this->kernel.setArg(a++, gVariance);
        this->kernel.setArg(a++, w);
        this->kernel.setArg(a++, h);
        this->kernel.setArg(a++, channels);

//        std::cout << "Kernel args " << ((double) (std::chrono::high_resolution_clock::now() - startDisp).count()) / 1e6 << std::endl;

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w / 8, h / 2), cl::NDRange(w / 8, 1));
//        queue.enqueueNDRangeKernel(this->kernel, cl::NullRange, cl::NDRange(frames[0].rows, frames[0].cols), cl::NDRange(frames[0].rows, 1));

//        std::cout << "Kernel enqueued " << ((double) (std::chrono::high_resolution_clock::now() - startDisp).count()) / 1e6 << std::endl;

        queue.finish();

//        std::cout << "Kernel done " << ((double) (std::chrono::high_resolution_clock::now() - startDisp).count()) / 1e6 << std::endl;

        queue.enqueueReadBuffer(gDisparity, CL_FALSE, 0, disparity.dataend - disparity.datastart, (void *)disparity.datastart);
        queue.enqueueReadBuffer(gVariance, CL_FALSE, 0, variance.dataend - variance.datastart, (void *)variance.datastart);
//        queue.enqueueReadBuffer(gQ, CL_FALSE, 0, sizeof(_q), (void *)&_q);
        queue.finish();
//
////        std::cout << "Memory read done " << ((double) (std::chrono::high_resolution_clock::now() - startDisp).count()) / 1e6 << std::endl;
//
//        cv::Mat element = getStructuringElement( cv::MorphShapes::MORPH_ELLIPSE, cv::Size( 5, 5 ), cv::Point( 2, 2 ) );
////        cv::dilate(disparity, disparity,element);
////        cv::boxFilter(disparity, disparity, -1, cv::Size(3, 3));

//        sgbm->compute(frames[0], frames[1], disparity);
//        cv::UMat uDisparity;

//        cv::UMat l, r;
//
//        cv::cvtColor(frames[0], l, cv::COLOR_RGB2GRAY);
//        cv::cvtColor(frames[1], r, cv::COLOR_RGB2GRAY);
//
//        l.flags &= ~64;
//        r.flags &= ~64;
//
//        bm->compute(l, r, disparity);
//
//        uDisparity.copyTo(disparity);
//        this->q = _q;
//        cv::Mat disp;
//
//        disparity.convertTo(disp, CV_32F);
//
////        cv::inpaint(disp, disp == 0, disp, 1, cv::INPAINT_NS);
//        disp.convertTo(disparity, disparity.type());
    }

    void DisparityEvaluator::lazyInitializeOcl() {
        if (this->oclInitialized) {
            return;
        }
        sgbm = cv::StereoSGBM::create(0, 384, 3, 0, 0, 1, 0, 0, 0, 0, cv::StereoSGBM::MODE_HH);
        bm = cv::StereoBM::create(384, 11);
        bm->setPreFilterType(cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
        bm->setUniquenessRatio(5);

        cl_int err = 0;
        if (!this->hasContext) {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);

            bool deviceFound = false;

            for (const auto & platform : platforms) {
                std::cout << "Found platform " << platform.getInfo<CL_PLATFORM_NAME>() << " vendor " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

                if (platforms.empty()){
                    throw std::runtime_error("No platforms found!");
                }

                for (const auto & dev : devices) {
                    unsigned int status = dev.getInfo<CL_DEVICE_AVAILABLE>();
                    std::cout << "\tFound device " << dev.getInfo<CL_DEVICE_NAME>() << " status " << status << std::endl;

                    if (status == 1) {
                        this->device = dev;
                        deviceFound = true;
                        break;
                    }
                }
            }

            if (!deviceFound) {
                throw std::runtime_error("No devices found!");
            }

            this->context = cl::Context(this->device, nullptr, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                std::cerr << "Create context failed " << err << std::endl;
                throw std::runtime_error("Create context failed");
            }
            this->hasContext = true;
        }

        std::ifstream kernelFile("src/cl/DisparityEvaluator.cl");
        std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

        this->program = cl::Program(this->context, src, false, &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Create program failed " << err << std::endl;
            throw std::runtime_error("Create program failed");
        }

        try {
            err = program.build("-Werror -w -cl-denorms-are-zero -cl-fast-relaxed-math -cl-finite-math-only -cl-unsafe-math-optimizations -cl-no-signed-zeros");
//            err = program.build("-cl-opt-disable -DDEBUG=1");
//            err = program.build("-Werror");
        } catch (...) {
            std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
                      << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw std::runtime_error("Build failed");
        }

        this->kernel = cl::Kernel(program, "DisparityEvaluator", &err);

        if (err != CL_SUCCESS) {
            std::cerr << "Create kernel failed " << err << std::endl;
            throw std::runtime_error("Create kernel failed");
        }

        this->queue = cl::CommandQueue(context, device);

        this->oclInitialized = true;
    }

    const char *DisparityEvaluator::openclErrorString(cl_int err) {
        switch (err) {
            case CL_SUCCESS:
                return "CL_SUCCESS";
            case CL_DEVICE_NOT_FOUND:
                return "CL_DEVICE_NOT_FOUND";
            case CL_DEVICE_NOT_AVAILABLE:
                return "CL_DEVICE_NOT_AVAILABLE";
            case CL_COMPILER_NOT_AVAILABLE:
                return "CL_COMPILER_NOT_AVAILABLE";
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case CL_OUT_OF_RESOURCES:
                return "CL_OUT_OF_RESOURCES";
            case CL_OUT_OF_HOST_MEMORY:
                return "CL_OUT_OF_HOST_MEMORY";
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case CL_MEM_COPY_OVERLAP:
                return "CL_MEM_COPY_OVERLAP";
            case CL_IMAGE_FORMAT_MISMATCH:
                return "CL_IMAGE_FORMAT_MISMATCH";
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case CL_BUILD_PROGRAM_FAILURE:
                return "CL_BUILD_PROGRAM_FAILURE";
            case CL_MAP_FAILURE:
                return "CL_MAP_FAILURE";
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
                return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case CL_COMPILE_PROGRAM_FAILURE:
                return "CL_COMPILE_PROGRAM_FAILURE";
            case CL_LINKER_NOT_AVAILABLE:
                return "CL_LINKER_NOT_AVAILABLE";
            case CL_LINK_PROGRAM_FAILURE:
                return "CL_LINK_PROGRAM_FAILURE";
            case CL_DEVICE_PARTITION_FAILED:
                return "CL_DEVICE_PARTITION_FAILED";
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
                return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            case CL_INVALID_VALUE:
                return "CL_INVALID_VALUE";
            case CL_INVALID_DEVICE_TYPE:
                return "CL_INVALID_DEVICE_TYPE";
            case CL_INVALID_PLATFORM:
                return "CL_INVALID_PLATFORM";
            case CL_INVALID_DEVICE:
                return "CL_INVALID_DEVICE";
            case CL_INVALID_CONTEXT:
                return "CL_INVALID_CONTEXT";
            case CL_INVALID_QUEUE_PROPERTIES:
                return "CL_INVALID_QUEUE_PROPERTIES";
            case CL_INVALID_COMMAND_QUEUE:
                return "CL_INVALID_COMMAND_QUEUE";
            case CL_INVALID_HOST_PTR:
                return "CL_INVALID_HOST_PTR";
            case CL_INVALID_MEM_OBJECT:
                return "CL_INVALID_MEM_OBJECT";
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case CL_INVALID_IMAGE_SIZE:
                return "CL_INVALID_IMAGE_SIZE";
            case CL_INVALID_SAMPLER:
                return "CL_INVALID_SAMPLER";
            case CL_INVALID_BINARY:
                return "CL_INVALID_BINARY";
            case CL_INVALID_BUILD_OPTIONS:
                return "CL_INVALID_BUILD_OPTIONS";
            case CL_INVALID_PROGRAM:
                return "CL_INVALID_PROGRAM";
            case CL_INVALID_PROGRAM_EXECUTABLE:
                return "CL_INVALID_PROGRAM_EXECUTABLE";
            case CL_INVALID_KERNEL_NAME:
                return "CL_INVALID_KERNEL_NAME";
            case CL_INVALID_KERNEL_DEFINITION:
                return "CL_INVALID_KERNEL_DEFINITION";
            case CL_INVALID_KERNEL:
                return "CL_INVALID_KERNEL";
            case CL_INVALID_ARG_INDEX:
                return "CL_INVALID_ARG_INDEX";
            case CL_INVALID_ARG_VALUE:
                return "CL_INVALID_ARG_VALUE";
            case CL_INVALID_ARG_SIZE:
                return "CL_INVALID_ARG_SIZE";
            case CL_INVALID_KERNEL_ARGS:
                return "CL_INVALID_KERNEL_ARGS";
            case CL_INVALID_WORK_DIMENSION:
                return "CL_INVALID_WORK_DIMENSION";
            case CL_INVALID_WORK_GROUP_SIZE:
                return "CL_INVALID_WORK_GROUP_SIZE";
            case CL_INVALID_WORK_ITEM_SIZE:
                return "CL_INVALID_WORK_ITEM_SIZE";
            case CL_INVALID_GLOBAL_OFFSET:
                return "CL_INVALID_GLOBAL_OFFSET";
            case CL_INVALID_EVENT_WAIT_LIST:
                return "CL_INVALID_EVENT_WAIT_LIST";
            case CL_INVALID_EVENT:
                return "CL_INVALID_EVENT";
            case CL_INVALID_OPERATION:
                return "CL_INVALID_OPERATION";
            case CL_INVALID_GL_OBJECT:
                return "CL_INVALID_GL_OBJECT";
            case CL_INVALID_BUFFER_SIZE:
                return "CL_INVALID_BUFFER_SIZE";
            case CL_INVALID_MIP_LEVEL:
                return "CL_INVALID_MIP_LEVEL";
            case CL_INVALID_GLOBAL_WORK_SIZE:
                return "CL_INVALID_GLOBAL_WORK_SIZE";
            case CL_INVALID_PROPERTY:
                return "CL_INVALID_PROPERTY";
            case CL_INVALID_IMAGE_DESCRIPTOR:
                return "CL_INVALID_IMAGE_DESCRIPTOR";
            case CL_INVALID_COMPILER_OPTIONS:
                return "CL_INVALID_COMPILER_OPTIONS";
            case CL_INVALID_LINKER_OPTIONS:
                return "CL_INVALID_LINKER_OPTIONS";
            case CL_INVALID_DEVICE_PARTITION_COUNT:
                return "CL_INVALID_DEVICE_PARTITION_COUNT";
            case CL_INVALID_PIPE_SIZE:
                return "CL_INVALID_PIPE_SIZE";
            case CL_INVALID_DEVICE_QUEUE:
                return "CL_INVALID_DEVICE_QUEUE";
            case CL_INVALID_SPEC_ID:
                return "CL_INVALID_SPEC_ID";
            case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
                return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
            default:
                return "UNDEFINED";
        }
    }
} // ecv