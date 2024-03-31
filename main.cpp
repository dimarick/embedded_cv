#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

void drawText(Mat & image, double fps);

int main(int argc, const char **argv)
{
    Mat image;
    VideoCapture capture;

    cerr << "Built with OpenCV " << CV_VERSION << endl;
#ifdef HAVE_OPENCV_HIGHGUI
    cv::namedWindow("Sample");
#endif

    capture.open(argv[1]);

    auto capFps = capture.get(cv::CAP_PROP_FPS);
    auto capWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    auto capHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto capPixFormat = (int)capture.get(cv::CAP_PROP_PVAPI_PIXELFORMAT);

    const char *humanPixelFormat;

    switch (capPixFormat) {
        case CAP_PVAPI_PIXELFORMAT_MONO8:
            humanPixelFormat = "Mono8";
            break;
        case CAP_PVAPI_PIXELFORMAT_MONO16:
            humanPixelFormat = "Mono16";
            break;
        case CAP_PVAPI_PIXELFORMAT_BAYER8:
            humanPixelFormat = "Bayer8";
            break;
        case CAP_PVAPI_PIXELFORMAT_BAYER16:
            humanPixelFormat = "Bayer16";
            break;
        case CAP_PVAPI_PIXELFORMAT_RGB24:
            humanPixelFormat = "Rgb24";
            break;
        case CAP_PVAPI_PIXELFORMAT_BGR24:
            humanPixelFormat = "Bgr24";
            break;
        case CAP_PVAPI_PIXELFORMAT_RGBA32:
            humanPixelFormat = "Rgba32";
            break;
        case CAP_PVAPI_PIXELFORMAT_BGRA32:
            humanPixelFormat = "Bgra32";
            break;
        default:
            humanPixelFormat = "n/a";
    }

    cerr << "Capture opened, " << capWidth << "x" << capHeight << "p" << capFps << "p, " << humanPixelFormat <<  ", " << capPixFormat << endl;

    auto prev = std::chrono::high_resolution_clock::now();

    if(capture.isOpened())
    {
        cerr << "Capture is opened" << endl;
        double fps = 0.;
        for(int i = 0;; i++)
        {
            capture.read(image);

            auto now = std::chrono::high_resolution_clock::now();
            auto us = (double)(now - prev).count();
            prev = now;

            if (fps <= 0.) {
                fps = 1e9 / us;
            } else {
                fps = 0.2 * 1e9 / us + 0.8 * fps;
            }

            if (image.empty()) {
                break;
            }

            drawText(image, fps);

            cerr << "fps " << fps << endl;

            cout.write((char *)image.data, image.dataend - image.datastart);
            cout.flush();

#ifdef HAVE_OPENCV_HIGHGUI
            imshow("Sample", image);

            if(waitKey(1) >= 0) {
                break;
            }
#endif
        }
    } else {
        cerr << "No capture" << endl;
        image = Mat::zeros(480, 640, CV_8UC1);
        drawText(image, 0);
#ifdef HAVE_OPENCV_HIGHGUI
        imshow("Sample", image);
        waitKey(0);
#endif
    }

    cerr << "exiting..." << endl;

    capture.release();
#ifdef HAVE_OPENCV_HIGHGUI
    cv::destroyAllWindows();
#endif

    return 0;
}

void drawText(Mat & image, double fps)
{
    char text[100];

    snprintf(text, 100, "Hello, Opencv. FPS %f", fps);

    putText(image, text,
            Point(20, 50),
            FONT_HERSHEY_COMPLEX, 1, // font face and scale
            Scalar(255, 255, 255), // white
            1, LINE_AA); // line thickness and type
}
