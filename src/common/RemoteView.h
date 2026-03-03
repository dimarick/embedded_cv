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
            RAW =  0x0,
            JPEG =  0x1,
        };

        struct CvMatHeader {
            CvMatTypeEnum type;
            CvMatCodecEnum codec;
            unsigned char channels;
            short x;
            short y;
            unsigned short w;
            unsigned short h;
            float scale;
        };

        struct ChannelSettings {
            std::string viewName;
            int channelId;
            short x;
            short y;
            short w;
            short h;
            float scale;
        };

        std::shared_ptr<mini_server::BroadcastingServer> server;
        std::thread serverThread;
        std::mutex channelsMutex;
        std::unordered_map<std::string, std::unordered_map<int, std::unordered_map<int, ChannelSettings>>> channelSettings;
        void initializeServer();
    public:
        void showMat(const std::string& viewName, const cv::Mat& mat);
        int waitKey();

        virtual ~RemoteView();

        std::vector<char> createMessageFromMat(const cv::Mat &mat, const cv::Rect &rect, float scale);
    };
}

#endif //EMBEDDED_CV_REMOTEVIEW_H
