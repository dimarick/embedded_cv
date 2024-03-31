#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>
#include <fcntl.h>
#include <opencv2/features2d.hpp>

void drawText(cv::Mat & image, double fps);

int main(int argc, const char **argv)
{
    cv::Mat image, target;
    cv::VideoCapture capture;

    std::cerr << (cv::ocl::haveOpenCL() ? "with OpenCl" : "cpu only") << std::endl;

    std::cerr << "Built with OpenCV " << CV_VERSION << std::endl;
#ifdef HAVE_OPENCV_HIGHGUI
    cv::namedWindow("Sample");
#endif

    capture.open(argv[1]);

    auto capFps = capture.get(cv::CAP_PROP_FPS);
    auto capWidth = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    auto capHeight = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto capPixFormat = (int)capture.get(cv::CAP_PROP_PVAPI_PIXELFORMAT);

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

    std::cerr << "Capture opened, " << capWidth << "x" << capHeight << "p" << capFps << "p, " << humanPixelFormat <<  ", " << capPixFormat << std::endl;

    auto prev = std::chrono::high_resolution_clock::now();

    capture.read(image);

    cv::Size targetSize(capWidth, capHeight);

    const char command[] = "ffmpeg -f rawvideo -pixel_format bgr24 -s %dx%d -re  -i - %s %s";

    size_t bufferSize = sizeof(command) + strlen(argv[2]) + strlen(argv[3]) + 50;
    char *formattedCommand = (char *)malloc(bufferSize);
    snprintf(formattedCommand, bufferSize, command, targetSize.width, targetSize.height, argv[2], argv[3]);

    std::cerr << formattedCommand << std::endl;
    std::cerr.flush();

    auto pipe = popen(formattedCommand, "w");

    if (pipe == nullptr) {
        std::cerr << "Failed to start " << formattedCommand << " errno " << strerror(errno) << std::endl;

        return -1;
    }

    cv::ocl::setUseOpenCL(true);

    auto orb = cv::ORB::create(200);
    std::vector<cv::KeyPoint> keyPoints;

    if(capture.isOpened())
    {
        std::cerr << "Capture is opened" << std::endl;
        double fps = 0.;
        for(int i = 0;; i++)
        {
            capture.read(image);

            auto now = std::chrono::high_resolution_clock::now();
            auto us = (double)(now - prev).count();
            prev = now;

            fps = 1e9 / us;

            if (image.empty()) {
                break;
            }

            orb->detect(image, keyPoints);

            for (auto k = keyPoints.begin(); k != keyPoints.end(); ++k) {
                cv::circle(image, k->pt, (int) k->size / 5, cv::Scalar(0, 0, 255), 1);
            }

            drawText(image, fps);

            std::cerr << "fps " << fps << std::endl;

            fwrite(image.data, sizeof(char), image.dataend - image.datastart, pipe);
            fflush(pipe);

#ifdef HAVE_OPENCV_HIGHGUI
            imshow("Sample", image);

            if(cv::waitKey(1) >= 0) {
                break;
            }
#endif
        }
    } else {
        std::cerr << "No capture" << std::endl;
        image = cv::Mat::zeros(480, 640, CV_8UC1);
        drawText(image, 0);
#ifdef HAVE_OPENCV_HIGHGUI
        imshow("Sample", image);
        cv::waitKey(0);
#endif
    }

    std::cerr << "exiting..." << std::endl;

    capture.release();
#ifdef HAVE_OPENCV_HIGHGUI
    cv::destroyAllWindows();
#endif

    free(formattedCommand);

    return 0;
}

void drawText(cv::Mat & image, double fps)
{
    char text[100];

    snprintf(text, 100, "Hello, Opencv. FPS %f", fps);

    putText(image, text,
            cv::Point(20, 50),
            cv::FONT_HERSHEY_COMPLEX, 1, // font face and scale
            cv::Scalar(255, 255, 255), // white
            1, cv::LINE_AA); // line thickness and type
}
