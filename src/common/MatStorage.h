#ifndef EMBEDDED_CV_MATSTORAGE_H
#define EMBEDDED_CV_MATSTORAGE_H

#include <string>
#include "opencv2/core.hpp"

namespace ecv {

    class MatStorage {
    public:
        static void matWrite(const std::string& filename, const cv::Mat& mat);
        static bool matRead(const std::string& filename, cv::Mat &mat);
        static void matWrite(const std::string &filename, const cv::UMat &mat);
        static bool matRead(const std::string &filename, cv::UMat &mat);
    };

} // ecv

#endif //EMBEDDED_CV_MATSTORAGE_H
