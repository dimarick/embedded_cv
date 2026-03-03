#ifndef EMBEDDED_CV_REMOTEVIEW_H
#define EMBEDDED_CV_REMOTEVIEW_H

#include <string>
#include <unordered_map>
#include <opencv.hpp>
#include <BroadcastingServer.h>
#include <thread>

namespace ecv {
    class RemoteView {
    private:
        enum CvMatTypeEnum : unsigned char {
            TYPE_8U =  0x10,
            TYPE_8S =  0x11,
            TYPE_16U = 0x20,
            TYPE_16S = 0x21,
            TYPE_16F = 0x22,
            TYPE_32F = 0x32,
            TYPE_64F = 0x42,
        };

        enum CvMatCodecEnum : unsigned char {
            JPEG =  0x0,
        };

        struct CvMatHeader {
            CvMatTypeEnum type;
            CvMatCodecEnum codec;
            unsigned char channels;
            unsigned short w;
            unsigned short h;
        };

        struct Channel {
            std::string name;
            std::shared_ptr<mini_server::BroadcastingServer> server;
            std::thread thread;
        };

        std::mutex channelsMutex;
        std::unordered_map<std::string, Channel> channels;
        const Channel &getOrCreateChannel(const std::string& name);
    public:
        void showMat(const std::string& viewName, const cv::Mat& mat);
        int waitKey();

        std::shared_ptr<CvMatHeader> createMessageFromMat(const cv::Mat &mat, size_t *size);

        virtual ~RemoteView();
    };
}

#endif //EMBEDDED_CV_REMOTEVIEW_H
