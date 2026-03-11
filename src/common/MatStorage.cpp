#include <fstream>
#include "MatStorage.h"

namespace ecv {
    void MatStorage::matWrite(const std::string &filename, const cv::Mat &mat) {
        std::ofstream fs(filename, std::fstream::binary);

        int type = mat.type();
        int channels = mat.channels();
        fs.write((char*)&mat.rows, sizeof(int));
        fs.write((char*)&mat.cols, sizeof(int));
        fs.write((char*)&type, sizeof(int));
        fs.write((char*)&channels, sizeof(int));

        if (mat.isContinuous()) {
            fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
        } else {
            int rowSize = CV_ELEM_SIZE(type) * mat.cols;
            for (int r = 0; r < mat.rows; ++r)
            {
                fs.write(mat.ptr<char>(r), rowSize);
            }
        }
    }

    bool MatStorage::matRead(const std::string &filename, cv::Mat &mat) {
        std::ifstream fs(filename, std::fstream::binary);

        if (!fs.good()) {
            mat = cv::Mat();
            return false;
        }

        // Header
        int rows, cols, type, channels;
        fs.read((char*)&rows, sizeof(int));
        fs.read((char*)&cols, sizeof(int));
        fs.read((char*)&type, sizeof(int));
        fs.read((char*)&channels, sizeof(int));

        // Data
        mat = cv::Mat(rows, cols, type);
        fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

        return true;
    }

    void MatStorage::matWrite(const std::string &filename, const cv::UMat &mat) {
        matWrite(filename, mat.getMat(cv::ACCESS_READ));
    }

    bool MatStorage::matRead(const std::string &filename, cv::UMat &mat) {
        cv::Mat tmp;
        if (!matRead(filename, tmp)) {
            return false;
        }

        tmp.copyTo(mat);

        return true;
    }
} // ecv