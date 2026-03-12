#ifndef EMBEDDED_CV_SOCKETCAPTURE_H
#define EMBEDDED_CV_SOCKETCAPTURE_H
#include "common/CaptureInfo.h"
#include "IpcServer.h"
#include "core/mat.hpp"
#include <atomic>
#include <condition_variable>
#include <shared_mutex>
#include <utility>

#include "SocketFactory.h"

namespace ecv {
    class SocketCapture {
        std::string captureSocketName;
        int captureSocket = -1;
        mini_server::IpcServer captureServer;

        std::shared_mutex readerLock;
        std::vector<cv::UMat> readingFrames;
        std::vector<CaptureInfo> frameInfo;
        std::thread captureThread;

        std::atomic<bool> hasNewFrames = false;
        std::mutex newFramesMutex;
        std::condition_variable newFramesAvailable;

    public:
        explicit SocketCapture(std::string captureSocketName) : captureSocketName(std::move(captureSocketName)) {}

        bool run() {
            auto onFrameSetReceived =
                [this](int socket, const void *buffer, size_t size) {
                    auto header = ecv::CaptureBuffer::getHeader(buffer, size);
                    if (header == nullptr) {
                        return;
                    }
                    auto info = header->getFirstCaptureInfo();
                    if (info == nullptr) {
                        return;
                    }
                    auto imageData = info->getImageData();
                    std::unique_lock lock(readerLock);

                    readingFrames.resize(header->nCaptures);
                    frameInfo.resize(header->nCaptures);

                    for (int i = 0; i < header->nCaptures; i++) {
                        frameInfo[i] = *info;
                        auto mat = cv::Mat(info->h, info->w, CV_8UC(info->channels), (void *)imageData);
                        mat.copyTo(readingFrames[i]);
                        info = info->getNextCaptureInfo();
                        imageData = info->getImageData();
                    }

                    hasNewFrames = true;
                    newFramesAvailable.notify_all();
            };

            captureSocket = mini_server::SocketFactory::createClientSocket(captureSocketName);

            if (captureSocket == -1 && errno == ECONNREFUSED) {
                return false;
            }

            captureServer.setSocket(captureSocket);
            captureServer.setOnMessage(onFrameSetReceived);
            captureThread = std::thread([this]() {captureServer.runClient();});

            return true;
        }

        void stop() {
            captureServer.stop();
            captureThread.join();

            if (captureSocket > 0) {
                close(captureSocket);
            }
        }

        bool isRunning() {
            return captureServer.isRunning();
        }

        std::vector<cv::UMat> getNewFrames(std::vector<CaptureInfo> *captureInfo = nullptr) {
            if (!hasNewFrames) {
                std::unique_lock lock(newFramesMutex);
                newFramesAvailable.wait_for(lock, std::chrono::milliseconds(200));

                if (!hasNewFrames) {
                    return {};
                }
            }

            std::vector<cv::UMat> frames(readingFrames.size());
            std::shared_lock lock(readerLock);
            frames.resize(readingFrames.size());
            for (int i = 0; i < readingFrames.size(); i++) {
                frames[i] = readingFrames[i].clone();
            }

            hasNewFrames = false;

            if (captureInfo != nullptr) {
                *captureInfo = frameInfo;
            }

            return frames;
        }
    };
};


#endif //EMBEDDED_CV_SOCKETCAPTURE_H