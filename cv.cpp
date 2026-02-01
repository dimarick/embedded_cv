#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#include "ImageProcessor.h"
#include "CvCommandHandler.h"
#include <iostream>
#include <chrono>
#include <cpptrace/cpptrace.hpp>
#include <CommandServer.h>
#include <BroadcastingServer.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <SocketFactory.h>
#include <csignal>
#include <core/opencl/ocl_defs.hpp>
#include <fstream>
#include <core/ocl.hpp>

#include "DisparityEvaluator.h"

using namespace mini_server;
using namespace cv;

static std::atomic running = true;

static BroadcastingServer broadcastingServer;
static BroadcastingServer streamingServer;
static CommandServer commandServer;

void invMap(const cv::Mat &src, cv::Mat &dest) {
    if (dest.empty()) {
        dest = Mat(src.size(), src.type());
    }

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            const auto &p = src.at<cv::Point2f>(y, x);
            int row = (int) std::round(p.y);
            int col = (int) std::round(p.x);
            auto &p2 = dest.at<cv::Point2f>(std::max(0, std::min(row, dest.rows - 1)), std::max(0, std::min(col, dest.cols - 1)));
            p2.x = (float)x;
            p2.y = (float)y;
        }
    }
}

void matread(const std::string& filename, Mat &mat)
{
    std::ifstream fs(filename, std::fstream::binary);

    if (!fs.good()) {
        mat = Mat();
        return;
    }

    // Header
    int rows, cols, type, channels;
    fs.read((char*)&rows, sizeof(int));         // rows
    fs.read((char*)&cols, sizeof(int));         // cols
    fs.read((char*)&type, sizeof(int));         // type
    fs.read((char*)&channels, sizeof(int));     // channels

    // Data
    mat = Mat(rows, cols, type);
    fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);
}

int main(int argc, const char **argv) {

    cv::Mat captureFrameLeft, captureFrameRight, readingLeft, readingRight, readingImLeft, readingImRight;
    cv::UMat output, outputLeft, inputLeft, imageLeft, outputRight, inputRight, imageRight;
    cv::Mat resultLeft, resultRight;
    cv::VideoCapture captureLeft, captureRight;
    cpptrace::register_terminate_handler();

    std::cerr << "Built with OpenCV " << CV_VERSION << std::endl;

    std::vector<int> params = {
            cv::VideoCaptureProperties::CAP_PROP_CONVERT_RGB, true,
            cv::VideoCaptureProperties::CAP_PROP_FPS, 30,
            cv::VideoCaptureProperties::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 1920,
            cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 1080,
            cv::VideoCaptureProperties::CAP_PROP_BUFFERSIZE, 10,
    };

    if (argc < 2) {
        std::cout << "too less command arguments";

        return -1;
    }

    if (strcmp(argv[1], argv[2]) == 0) {
        captureLeft.open(argv[1], cv::CAP_V4L2, params);
        captureRight = captureLeft;
    } else {
        captureLeft.open(argv[1], cv::CAP_V4L2, params);
        captureRight.open(argv[2], cv::CAP_V4L2, params);
    }

    broadcastingServer.setSocket(SocketFactory::createListeningSocket("/tmp/cv_tm", 10));
    streamingServer.setSocket(SocketFactory::createListeningSocket("/tmp/cv_s", 10));
    commandServer.setSocket(SocketFactory::createListeningSocket("/tmp/cv_ctl", 1));
    auto handler = CvCommandHandler(commandServer);
    commandServer.setHandler(handler);

    signal(SIGINT, [](int signal) {
        running = false;
        broadcastingServer.stop();
        commandServer.stop();
    });

    std::thread commandServerThread = std::thread([]() {
        commandServer.run();
    });

    std::thread broadcastingServerThread = std::thread([]() {
        broadcastingServer.run();
    });

    std::thread streamingServerThread = std::thread([]() {
        streamingServer.run();
    });

    auto capFps = captureLeft.get(cv::CAP_PROP_FPS);
    auto capWidth = (int) captureLeft.get(cv::CAP_PROP_FRAME_WIDTH);
    auto capHeight = (int) captureLeft.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto capPixFormat = (int) captureLeft.get(cv::CAP_PROP_PVAPI_PIXELFORMAT);
    auto capBuffer = (int) captureLeft.get(cv::CAP_PROP_BUFFERSIZE);

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
              << ", " << capPixFormat << ", buffer " << capBuffer << std::endl;
    std::cerr.flush();

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

    captureLeft.release();
    captureRight.release();

    cv::Size outputSize(capWidth, capHeight);
    cv::Size processingSize(capWidth, capHeight);

    const char inputCommand[] = "ffmpeg -loglevel fatal -f v4l2 -input_format mjpeg -s %dx%d -re  -i %s -f rawvideo -filter:v 'format=bgr24' -fflags nobuffer -avioflags direct -fflags discardcorrupt -g 15 -threads 7 -";

    size_t inputBufferSize = sizeof(inputCommand) + strlen(argv[1]) + strlen(argv[2]) + 50;
    char *formattedInputCommand = (char *) malloc(inputBufferSize);
    snprintf(formattedInputCommand, inputBufferSize, inputCommand, capWidth, capHeight, argv[1]);

    std::cerr << formattedInputCommand << std::endl;
    std::cerr.flush();

    auto leftPipe = popen(formattedInputCommand, "r");

    if (leftPipe == nullptr) {
        std::cerr << "Failed to start " << formattedInputCommand << " errno " << strerror(errno) << std::endl;

        free(formattedInputCommand);
        return -1;
    }

    snprintf(formattedInputCommand, inputBufferSize, inputCommand, capWidth, capHeight, argv[2]);

    std::cerr << formattedInputCommand << std::endl;
    std::cerr.flush();

    auto rightPipe = popen(formattedInputCommand, "r");

    if (rightPipe == nullptr) {
        std::cerr << "Failed to start " << formattedInputCommand << " errno " << strerror(errno) << std::endl;

        free(formattedInputCommand);
        return -1;
    }

    free(formattedInputCommand);

    ImageProcessor processor(processingSize.width, processingSize.height, broadcastingServer);

    handler.setImageProcessor(&processor);

    double fps = 0.;
    double avgFps = 0.;
    double avgTime = 0.;
    double avgTime2 = 0.;

    captureFrameLeft.copyTo(readingLeft);
    captureFrameRight.copyTo(readingRight);

    std::thread writer;
    std::atomic writerIsRunning = false;

    std::thread imageWriter;
    std::atomic imageWriterIsRunning = false;

    std::mutex readerLeftLock;
    std::mutex readerRightLock;
    std::atomic readerLeftCount = 0l;
    std::atomic readerRightCount = 0l;
    auto frameCount = 0l;

    auto readerfunction =
            [](cv::Mat *reading, FILE *pipe, cv::Mat *readingIm, std::mutex *lock, std::atomic<long> *count) {
                while (running) {
                    fread(reading->data, sizeof(char), reading->dataend - reading->datastart, pipe);
                    lock->lock();
                    reading->copyTo(*readingIm);
                    (*count)++;
                    lock->unlock();
                }
            };
    auto readerLeft = std::thread(readerfunction, &readingLeft, leftPipe, &readingImLeft, &readerLeftLock, &readerLeftCount);
    auto readerRight = std::thread(readerfunction, &readingRight, rightPipe, &readingImRight, &readerRightLock, &readerRightCount);

    ecv::DisparityEvaluator disparityEvaluator;

    disparityEvaluator.lazyInitializeOcl();
#ifndef HAVE_OPENCL
        throw std::runtime_error("OpenCV built without opencl");
#endif

    if (!cv::ocl::isOpenCLActivated()) {
        throw std::runtime_error("OpenCL is not available");
    }

    UMat calibLeft, calibRight;

    UMat lFrame, rFrame;
    UMat lDispMap, rDispMap, leftDispMap, rightDispMap;

    std::vector<cv::UMat> frames(2);
    cv::Mat aligned;
    cv::Mat alignedMap;
    cv::Mat invAlignedMap;
    std::vector<cv::Mat> bestMap1(frames.size()), bestMap2(frames.size());
    std::vector<cv::Mat> invBestMap1(frames.size()), invBestMap2(frames.size());

    matread("map0.bin", bestMap1[0]);
    matread("map1.bin", bestMap1[1]);
    matread("mapa.bin", alignedMap);

    for (int i = 0; i < frames.size(); ++i) {
        invMap(bestMap1[i], invBestMap1[i]);
    }
    invMap(alignedMap, invAlignedMap);

#ifdef HAVE_OPENCV_HIGHGUI
    cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("src " + std::to_string(0), cv::WINDOW_AUTOSIZE);

    auto mouseDisp = cv::Point2i();
    auto mouseSrc = cv::Point2i();

    void (*onMouse)(int, int, int, int, void *) = [](int event, int x, int y, int flags, void *userdata) {
        auto mouse = (cv::Point2i *) userdata;
        mouse->x = x;
        mouse->y = y;
    };
    cv::setMouseCallback("Disparity", onMouse, &mouseDisp);
    cv::setMouseCallback("src " + std::to_string(0), onMouse, &mouseSrc);
#endif

//    uint8_t im1[]  = {0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0};
//    uint8_t im2[]  = {0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0};
//    int16_t disp[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//    std::vector<cv::Mat> test = {cv::Mat(1, 20, CV_8U, im1), cv::Mat(1, 20, CV_8U, im2)};
//    cv::Mat dispMat = cv::Mat(1, 20, CV_16S, disp);
//
//    disparityEvaluator.evaluateDisparity(test, dispMat);

    double minVal = 0, maxVal = 0, varianceMinVal = 0, varianceMaxVal = 0;

    cv::Mat disparity8;

    Mat preview;
    std::mutex sendingLock;
    std::ostringstream tm;
    std::vector<cv::UMat> result(frames.size());
    cv::FileStorage fs;
    cv::FileStorage fs_write;
    cv::Mat kernel;
    float kernelDiv = 1;
    auto clahe = cv::createCLAHE(1, cv::Size(3,3));


    for (int i = 0; running; i++) {
        long nextFrame = std::max(readerLeftCount, readerRightCount);
        if (readerLeftCount == frameCount || readerRightCount == frameCount || readingImLeft.empty() || readingImRight.empty()) {
            usleep(1);
            continue;
        }

        frameCount = nextFrame;

        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::system_clock::time_point startDisp;
        std::chrono::system_clock::time_point endDisp;

        readerLeftLock.lock();
        readingImLeft.copyTo(imageLeft);
        readerLeftLock.unlock();

        readerRightLock.lock();
        readingImRight.copyTo(imageRight);
        readerRightLock.unlock();

        cv::rotate(imageRight, imageRight, ROTATE_180);

        frames = {imageLeft, imageRight};

        auto now = std::chrono::high_resolution_clock::now();
        auto us = (double) (now - prev).count();
        prev = now;

        fps = 1e9 / us;

        if (imageLeft.empty()) {
            break;
        }

        for (int j = 0; j < frames.size(); ++j) {
//            for (int k = 1; k < 12; ++k) {
//                auto p1 = Point(0, k * frames[j].size().height / 11);
//                auto p2 = Point(frames[j].size().width, k * frames[j].size().height / 11);
////                    cv::line(frames[j], p1, p2, cv::Scalar(255, 0, 255), 1);
//            }

            kernel.release();

            if (!fs.isOpened()) {
                fs.open("kernel.yml", cv::FileStorage::READ);
            }

            if (fs.isOpened()) {
                fs["kernel"] >> kernel;
                fs["kernel_div"] >> kernelDiv;
                fs.release();

                kernel /= kernelDiv;
            } else {
                std::cout << "Файл kernel.yml не найден, создаю дефолтное ядро 5x5" << std::endl;
                kernel = cv::Mat::ones(5, 5, CV_32F);

                fs_write.open("kernel.yml", cv::FileStorage::WRITE);
                fs_write << "kernel" << kernel << "kernel_div" << 25;
                fs_write.release();
            }

#ifdef HAVE_OPENCV_HIGHGUI
            cv::UMat filtered;
            cv::UMat original;
            frames[0].copyTo(original);

            cv::cvtColor(frames[0], original, cv::COLOR_RGB2GRAY);
//
//            cv::filter2D(original, filtered, kernel);
//            cv::medianBlur(original, filtered, 17);
//            cv::medianBlur(original, filtered, 5);
//            cv::equalizeHist(filtered, filtered);
            clahe->apply(original, filtered);
            clahe->apply(filtered, filtered);

            imshow("Original", original);
            imshow("Filtered", filtered);

            if (j != 1) {
                cv::remap(frames[j], result[j], bestMap1[j], noArray(), INTER_NEAREST);
            } else {
                cv::remap(frames[1], result[j], alignedMap, noArray(), INTER_NEAREST);
            }

            for (int k = 1; k < 12; ++k) {
                auto p1 = Point(0, k * frames[j].size().height / 11);
                auto p2 = Point(frames[j].size().width, k * frames[j].size().height / 11);
//                    cv::line(result[j], p1, p2, cv::Scalar(0, 255, 255), 1);
            }

#endif
        }

#ifdef HAVE_OPENCV_HIGHGUI
        std::vector<cv::Mat> result2(result.size());
        for (auto j = 0; j < result2.size(); j++) {
            result[j].getMat(ACCESS_RW).copyTo(result2[j]);
        }
        cv::drawMarker(frames[0], mouseSrc, cv::Scalar(255, 0, 255), MarkerTypes::MARKER_CROSS, 30, 2);
        auto plainMap1 = invBestMap1[0].ptr<cv::Point2f>(mouseSrc.y, mouseSrc.x);
        auto plainMap2 = alignedMap.ptr<cv::Point2f>((int)plainMap1->y, (int)plainMap1->x);
        cv::drawMarker(result2[0], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), MarkerTypes::MARKER_TILTED_CROSS, 30, 2);
        cv::drawMarker(frames[1], cv::Point2i((int) plainMap2->x, (int) plainMap2->y), cv::Scalar(0, 255, 0), MarkerTypes::MARKER_CROSS, 30, 2);
        cv::drawMarker(result2[1], cv::Point2i((int) plainMap1->x, (int) plainMap1->y), cv::Scalar(0, 255, 0), MarkerTypes::MARKER_TILTED_CROSS, 30, 2);

        cv::Mat varianceFp;
        cv::Mat variance8;
#endif
        cv::Mat disparityFp;
        cv::Mat disparity;
        cv::Mat variance;
        disparity.setTo(0);
//            variance.setTo(0);

        startDisp = std::chrono::high_resolution_clock::now();
        disparityEvaluator.evaluateDisparity(result, disparity, variance);
        endDisp = std::chrono::high_resolution_clock::now();

        disparity.copyTo(disparityFp);
        if (minVal == 0 || maxVal == 0) {
            cv::minMaxLoc(disparityFp, &minVal, &maxVal);
            maxVal = 300 * ecv::DisparityEvaluator::DISPARITY_PRECISION;
            minVal = 0;
        }

        disparityFp -= minVal;
        disparityFp *= 255.0 / (maxVal - minVal);


        disparityFp.convertTo(disparity8, CV_8U);
        cv::applyColorMap(disparity8, disparity8, ColormapTypes::COLORMAP_JET);
#ifdef HAVE_OPENCV_HIGHGUI
        variance.copyTo(varianceFp);
        variance.copyTo(varianceFp);

        cv::minMaxLoc(variance, &varianceMinVal, &varianceMaxVal);

        varianceFp -= varianceMinVal;
        varianceFp *= 255.0 / (varianceMaxVal - varianceMinVal);

        varianceFp.convertTo(variance8, CV_8U);
        cv::applyColorMap(variance8, variance8, ColormapTypes::COLORMAP_JET);

        cv::drawMarker(disparity8, mouseDisp, cv::Scalar(255, 128, 255), MarkerTypes::MARKER_CROSS, 30, 3);
        cv::drawMarker(result2[0], mouseDisp, cv::Scalar(255, 128, 255), MarkerTypes::MARKER_CROSS, 30, 3);
        cv::drawMarker(result2[1], mouseDisp, cv::Scalar(255, 128, 255), MarkerTypes::MARKER_CROSS, 30, 3);
        cv::drawMarker(variance8, mouseDisp, cv::Scalar(255, 128, 255), MarkerTypes::MARKER_CROSS, 30, 3);
        auto disparityAtPoint = disparity.at<int16_t>(mouseDisp.y, mouseDisp.x);
        auto varianceAtPoint = variance.at<float>(mouseDisp.y, mouseDisp.x);
        auto dispStr = std::to_string((float)disparityAtPoint / ecv::DisparityEvaluator::DISPARITY_PRECISION);
        auto varStr = std::to_string((float)varianceAtPoint);
        cv::putText(disparity8, dispStr, mouseDisp, FONT_HERSHEY_COMPLEX, 3, cv::Scalar(255, 192, 255));
        cv::putText(variance8, varStr, mouseDisp, FONT_HERSHEY_COMPLEX, 3, cv::Scalar(255, 192, 255));
        cv::imshow("Disparity", disparity8);
        cv::imshow("Variance", variance8);
        for (int j = 0; j < result2.size(); ++j) {
            cv::imshow("Plain best " + std::to_string(j), result2[j]);
//                imshow("src " + std::to_string(j), frames[j]);
        }

#endif

        auto end = std::chrono::high_resolution_clock::now();

        double time = ((double) (end - start).count()) / 1e6;
        double time2 = ((double) (endDisp - startDisp).count()) / 1e6;
        double avgA = 2. / ((i < 50 ? 5 : 50) + 1);
        avgTime = avgTime == 0. ? time : avgA * time + (1 - avgA) * avgTime;
        avgTime2 = avgTime2 == 0. ? time2 : avgA * time2 + (1 - avgA) * avgTime2;
        avgFps = avgFps == 0. ? fps : avgA * fps + (1 - avgA) * avgFps;

        std::cerr << "fps " << avgFps << " time " << time << " time2 " << time2 << " load " << avgTime * 100 / ((double)us / 1e6) << " %, avg " << avgTime << " %, avg2 " << avgTime2 << " size "
                  << resultLeft.dataend - resultLeft.datastart << std::endl;

        sendingLock.lock();
        tm << "PERF " << fps << " " << time << std::endl;
        cv::resize(disparity8, preview, outputSize);
        sendingLock.unlock();

        if (!writerIsRunning) {
            if (writer.joinable()) {
                writer.join();
            }

            writerIsRunning = true;

            writer = std::thread([](auto *isRunning, std::ostringstream *tm, auto *sendingLock) {
                sendingLock->lock();
                const std::string message = tm->str();
                tm->str("");
                tm->clear();
                sendingLock->unlock();

                broadcastingServer.broadcast(message);
                *isRunning = false;
            }, &writerIsRunning, &tm, &sendingLock);
        }

        if (!imageWriterIsRunning) {
            if (imageWriter.joinable()) {
                imageWriter.join();
            }

            imageWriterIsRunning = true;

            imageWriter = std::thread([](auto *preview, auto *isRunning, auto *sendingLock) {
                sendingLock->lock();
                Mat frame;
                preview->copyTo(frame);
                sendingLock->unlock();

                std::vector<uchar> jpgData;

                cv::imencode(".jpg", frame, jpgData, {
                        ImwriteFlags::IMWRITE_JPEG_QUALITY, 70,
                        ImwriteFlags::IMWRITE_JPEG_OPTIMIZE, 1,
                        ImwriteFlags::IMWRITE_JPEG_SAMPLING_FACTOR, cv::ImwriteJPEGSamplingFactorParams::IMWRITE_JPEG_SAMPLING_FACTOR_420,
                });

                std::cerr << "Sending jpeg size " << jpgData.size() << "/" << frame.dataend - frame.datastart << "(" << (float)jpgData.size() / (float)(frame.dataend - frame.datastart) << ")" << std::endl;

                streamingServer.broadcast(jpgData.data(), jpgData.size(), 100);
                *isRunning = false;
            }, &preview, &imageWriterIsRunning, &sendingLock);
        }


#ifdef HAVE_OPENCV_HIGHGUI

        if (cv::waitKey(1) >= 0) {
            break;
        }
#endif
    }

    if (writer.joinable()) {
        writer.join();
    }

    readerLeft.join();
    readerRight.join();

    std::cerr << "exiting..." << std::endl;

    captureLeft.release();
    captureRight.release();

#ifdef HAVE_OPENCV_HIGHGUI
    cv::destroyAllWindows();
#endif

    broadcastingServerThread.join();
    commandServerThread.join();

    return 0;
}
