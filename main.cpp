#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include "opencv2/videoio.hpp"
#include "ImageProcessor.h"
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>
#include <cpptrace/cpptrace.hpp>

void drawText(cv::UMat &image, double fps);

int main(int argc, const char **argv) {
    cv::Mat captureFrameLeft, captureFrameRight;
    cv::UMat output, outputLeft, inputLeft, imageLeft, outputRight, inputRight, imageRight;
    cv::Mat resultLeft, resultRight;
    cv::VideoCapture captureLeft, captureRight;

    cpptrace::register_terminate_handler();

    std::cerr << (cv::ocl::haveOpenCL() ? "with OpenCl" : "cpu only") << std::endl;

    std::cerr << "Built with OpenCV " << CV_VERSION << ", cv::ocl::haveSVM(): " << cv::ocl::haveSVM() << std::endl;

    captureLeft.open(argv[1]);
    captureRight.open(argv[2]);

    auto capFps = captureLeft.get(cv::CAP_PROP_FPS);
    auto capWidth = (int) captureLeft.get(cv::CAP_PROP_FRAME_WIDTH);
    auto capHeight = (int) captureLeft.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto capPixFormat = (int) captureLeft.get(cv::CAP_PROP_PVAPI_PIXELFORMAT);

    const char *humanPixelFormat;

    switch (capPixFormat) {
        case cv::CAP_PVAPI_PIXELFORMAT_MONO8:
            humanPixelFormat = "Mono8";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_MONO16:
            humanPixelFormat = "Mono16";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BAYER8:
            humanPixelFormat = "Bayer8";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BAYER16:
            humanPixelFormat = "Bayer16";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_RGB24:
            humanPixelFormat = "Rgb24";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BGR24:
            humanPixelFormat = "Bgr24";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_RGBA32:
            humanPixelFormat = "Rgba32";
            break;
        case cv::CAP_PVAPI_PIXELFORMAT_BGRA32:
            humanPixelFormat = "Bgra32";
            break;
        default:
            humanPixelFormat = "n/a";
    }

    std::cerr << "Capture opened, " << capWidth << "x" << capHeight << "p" << capFps << "p, " << humanPixelFormat
              << ", " << capPixFormat << std::endl;

    auto prev = std::chrono::high_resolution_clock::now();

    if (!captureLeft.isOpened()) {
        std::cerr << "Left Capture is not opened" << std::endl;
        return -1;
    }

    if (!captureRight.isOpened()) {
        std::cerr << "Right Capture is not opened" << std::endl;
        return -1;
    }

    captureLeft.read(captureFrameLeft);
    captureRight.read(captureFrameRight);

    cv::Size outputSize(capWidth, capHeight);
    cv::Size processingSize(capWidth, capHeight);

    const char command[] = "ffmpeg -f rawvideo -pixel_format bgr24 -s %dx%d -re  -i - %s %s";

    size_t bufferSize = sizeof(command) + strlen(argv[3]) + strlen(argv[4]) + 50;
    char *formattedCommand = (char *) malloc(bufferSize);
    snprintf(formattedCommand, bufferSize, command, outputSize.width, outputSize.height, argv[3], argv[4]);

    std::cerr << formattedCommand << std::endl;
    std::cerr.flush();

    auto pipe = popen(formattedCommand, "w");

    free(formattedCommand);

    if (pipe == nullptr) {
        std::cerr << "Failed to start " << formattedCommand << " errno " << strerror(errno) << std::endl;

        return -1;
    }

    ImageProcessor processor(processingSize.width, processingSize.height);

    double fps = 0.;
    double avgTime = 0.;

    for (int i = 0;; i++) {
        captureLeft.read(captureFrameLeft);
        captureRight.read(captureFrameRight);
        captureFrameLeft.copyTo(inputLeft);
        captureFrameRight.copyTo(inputRight);

        cv::ocl::setUseOpenCL(true);

        auto now = std::chrono::high_resolution_clock::now();
        auto us = (double) (now - prev).count();
        prev = now;

        fps = 1e9 / us;

        if (inputLeft.empty()) {
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        cv::resize(inputLeft, imageLeft, processingSize, 0, 0,
                   cv::INTER_NEAREST);

        cv::resize(inputRight, imageRight, processingSize, 0, 0,
                   cv::INTER_NEAREST);

        processor.processFrame(imageLeft, imageRight, output);

        drawText(imageLeft, fps);

        cv::resize(imageLeft, outputLeft, outputSize, 0, 0,
                   cv::INTER_NEAREST);

        cv::resize(imageRight, outputRight, outputSize, 0, 0,
                   cv::INTER_NEAREST);

        auto end = std::chrono::high_resolution_clock::now();

        cv::ocl::setUseOpenCL(false);

        outputLeft.copyTo(resultLeft);
        outputRight.copyTo(resultRight);

        double time = ((double) (end - start).count()) / 1e6;
        double avgA = 2. / ((i < 50 ? 50 : 500) + 1);
        avgTime = avgTime == 0. ? time : avgA * time + (1 - avgA) * avgTime;

        std::cerr << "fps " << fps << " time " << time << " avg " << avgTime << " size "
                  << resultLeft.dataend - resultLeft.datastart << std::endl;

//        fwrite(resultLeft.data, sizeof(char), resultLeft.dataend - resultLeft.datastart, pipe);
//        fflush(pipe);

#ifdef HAVE_OPENCV_HIGHGUI
        imshow("Left", resultLeft);
//        imshow("Right", resultRight);
        if (!output.empty()) {
            imshow("Result", output);
        }

        if (cv::waitKey(1) >= 0) {
            break;
        }
#endif
    }
    std::cerr << "exiting..." << std::endl;

    captureLeft.release();
    captureRight.release();

#ifdef HAVE_OPENCV_HIGHGUI
    cv::destroyAllWindows();
#endif

    return 0;
}

void drawText(cv::UMat &image, double fps) {
    char text[100];

    snprintf(text, 100, "Hello, Opencv. FPS %f", fps);

    cv::putText(image, text,
                cv::Point(20, 50),
                cv::FONT_HERSHEY_COMPLEX, 1, // font face and scale
                cv::Scalar(255, 255, 255), // white
                1, cv::LINE_AA); // line thickness and type
}
