#ifndef EMBEDDED_CV_REMOTEVIEW_H
#define EMBEDDED_CV_REMOTEVIEW_H

#include <string>
#include <unordered_map>
#include <opencv.hpp>
#include <IpcServer.h>
#include <ranges>
#include <shared_mutex>
#include <thread>

namespace ecv {
    class RemoteView {
    private:
        enum CvMatTypeEnum : char {
            TYPE_8U =  0x10,
            TYPE_8S =  0x11,
            TYPE_16U = 0x20,
            TYPE_16S = 0x21,
            TYPE_16F = 0x22,
            TYPE_32F = 0x32,
            TYPE_64F = 0x42,
        };

        enum CvMatCodecEnum : char {
            RAW =  0x0,
            JPEG =  0x1,
        };

        struct StringHeader {
            short nameSize;
        };

        struct CvMatHeader {
            CvMatTypeEnum type;
            CvMatCodecEnum codec;
            short channels;
            short viewW;
            short viewH;
            short x;
            short y;
            short w;
            short h;
        };

        struct ChannelSettings {
            std::string viewName;
            int channelId = -1;
            short viewW = 0;
            short viewH = 0;
            short x = 0;
            short y = 0;
            short w = 0;
            short h = 0;
        };

        const std::string &socketPath;
        std::shared_ptr<mini_server::IpcServer> server;
        std::thread serverThread;
        std::mutex channelsMutex;
        std::unordered_map<std::string, std::unordered_map<int, std::unordered_map<int, ChannelSettings>>> channelSettings;
        mutable std::shared_mutex viewsMutex;
        std::unordered_map<std::string, CvMatHeader> views;
        void initializeServer();
    public:
        explicit RemoteView(const std::string &socketPath) : socketPath(socketPath) {}

        void showMat(const std::string& viewName, const cv::Mat& mat);
        int waitKey();

        CvMatHeader createMessageHeaderFromMat(const cv::Mat &mat, const cv::Rect &rect, short viewW, short viewH);

        virtual ~RemoteView();

        std::vector<char> createMessageFromMat(CvMatHeader header, const cv::Mat &mat, const cv::Rect &rect, short viewW, short viewH);

        void showMat(const std::string &viewName, const cv::UMat &mat);

        std::unordered_map<std::string, CvMatHeader> getViews() const {
            decltype(views )tmp;
            {
                std::shared_lock lock(viewsMutex);
                tmp = views;
            }
            return tmp;
        }
    };
}

#endif //EMBEDDED_CV_REMOTEVIEW_H
