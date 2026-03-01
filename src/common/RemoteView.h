#ifndef EMBEDDED_CV_REMOTEVIEW_H
#define EMBEDDED_CV_REMOTEVIEW_H

#include <string>
#include <opencv.hpp>

namespace ecv {
    class RemoteView {
    public:
        static void showMat(const std::string& viewName, const cv::Mat& mat);
    };
}

#endif //EMBEDDED_CV_REMOTEVIEW_H
