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
    cv::Mat captureFrame;
    cv::UMat output, input, image;
    cv::VideoCapture capture;

    cpptrace::register_terminate_handler();

    std::cerr << (cv::ocl::haveOpenCL() ? "with OpenCl" : "cpu only") << std::endl;

    std::cerr << "Built with OpenCV " << CV_VERSION << ", cv::ocl::haveSVM(): " << cv::ocl::haveSVM() << std::endl;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::namedWindow("Sample");
#endif

    capture.open(argv[1]);

    capture.set(cv::CAP_PROP_CONVERT_RGB, false);
    auto capFps = capture.get(cv::CAP_PROP_FPS);
    auto capWidth = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
    auto capHeight = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto capPixFormat = (int) capture.get(cv::CAP_PROP_PVAPI_PIXELFORMAT);

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

    capture.read(captureFrame);

    captureFrame.type();

    cv::Size outputSize(640, 480);
    cv::Size processingSize(640, 480);

    const char command[] = "ffmpeg -f rawvideo -pixel_format yuyv422 -s %dx%d -re  -i - %s %s";

    size_t bufferSize = sizeof(command) + strlen(argv[2]) + strlen(argv[3]) + 50;
    char *formattedCommand = (char *) malloc(bufferSize);
    snprintf(formattedCommand, bufferSize, command, outputSize.width, outputSize.height, argv[2], argv[3]);

    std::cerr << formattedCommand << std::endl;
    std::cerr.flush();

    auto pipe = popen(formattedCommand, "w");

    free(formattedCommand);

    if (pipe == nullptr) {
        std::cerr << "Failed to start " << formattedCommand << " errno " << strerror(errno) << std::endl;

        return -1;
    }

    cv::Mat result, colorResult;

    ImageProcessor processor(processingSize.width, processingSize.height);

    if (capture.isOpened()) {
        std::cerr << "Capture is opened" << std::endl;
        double fps = 0.;
        double avgTime = 0.;

        for (int i = 0;; i++) {
            capture.read(captureFrame);
            captureFrame.copyTo(input);

            cv::ocl::setUseOpenCL(true);

            auto now = std::chrono::high_resolution_clock::now();
            auto us = (double) (now - prev).count();
            prev = now;

            fps = 1e9 / us;

            if (input.empty()) {
                break;
            }

            auto start = std::chrono::high_resolution_clock::now();


            cv::resize(input, image, processingSize, 0, 0,
                       cv::INTER_NEAREST);

            processor.processFrame(image);

            drawText(image, fps);

            cv::resize(image, output, outputSize, 0, 0,
                       cv::INTER_NEAREST);

            auto end = std::chrono::high_resolution_clock::now();

            cv::ocl::setUseOpenCL(false);

            output.copyTo(result);

            double time = ((double) (end - start).count()) / 1e6;
            double avgA = 2. / ((i < 50 ? 50 : 500) + 1);
            avgTime = avgTime == 0. ? time : avgA * time + (1 - avgA) * avgTime;

            std::cerr << "fps " << fps << " time " << time << " avg " << avgTime << " size "
                      << result.dataend - result.datastart << std::endl;

            fwrite(result.data, sizeof(char), result.dataend - result.datastart, pipe);
            fflush(pipe);

#ifdef HAVE_OPENCV_HIGHGUI
            cv::cvtColor(result, colorResult, cv::COLOR_YUV2BGR_YUYV);
            imshow("Sample", colorResult);

            if (cv::waitKey(1) >= 0) {
                break;
            }
#endif
        }
    } else {
        std::cerr << "No capture" << std::endl;

        capture.release();

        return 0;
    }
    std::cerr << "exiting..." << std::endl;

    capture.release();

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
