#ifndef EMBEDDED_CV_SOCKETCAPTURE_H
#define EMBEDDED_CV_SOCKETCAPTURE_H
#include "common/CaptureInfo.h"
#include "IpcServer.h"
#include "core/mat.hpp"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <shared_mutex>
#include <utility>

#include "SocketFactory.h"

namespace ecv {
    class SocketCapture {
        std::string captureSocketName;
        int captureSocket = -1;
        mini_server::IpcServer captureServer;

        std::shared_mutex readerLock;
        std::vector<cv::Mat> readingFrames;
        std::vector<CaptureInfo> frameInfo;
        std::thread captureThread;

        std::atomic<bool> hasNewFrames = false;
        std::mutex newFramesMutex;
        std::condition_variable newFramesAvailable;

        int reconnect() {
            while (true) {
                std::cerr << "Try connecting to " << captureSocketName << "..." << strerror(errno) << std::endl;
                captureSocket = mini_server::SocketFactory::createClientSocket(captureSocketName);

                if (captureSocket == -1 && (errno == ECONNREFUSED || errno == ENOENT)) {
                    sleep(1);
                    continue;
                }
                break;
            }

            std::cerr << "Connected to " << captureSocketName << "!" << std::endl;

            return captureSocket;
        };

    public:
        explicit SocketCapture(std::string captureSocketName) : captureSocketName(std::move(captureSocketName)) {}

        void run() {
            auto onFrameSetReceived =
                [this](int socket, const void *buffer, size_t size) {
                    auto bufferEnd = reinterpret_cast<const char *>(buffer) + size;
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
                        if (info->w * info->h * info->channels != info->size) {
                            // throw std::runtime_error(std::format("Failed capture info size {}*{}*{}={}", info->w, info->h, info->channels, info->size));
                            return;
                        }

                        auto captureEnd = imageData + info->size;
                        if (captureEnd > bufferEnd) {
                            // throw std::runtime_error(std::format("Buffer overflow {} {}", captureEnd, bufferEnd));
                            return;
                        }

                        auto mat = cv::Mat(info->h, info->w, CV_8UC(info->channels), (void *)imageData);
                        mat.copyTo(readingFrames[i]);
                        if (i < header->nCaptures - 1) {
                            info = info->getNextCaptureInfo();
                            imageData = info->getImageData();
                        }
                    }

                    hasNewFrames = true;
                    newFramesAvailable.notify_all();
            };

            captureServer.setOnMessage(onFrameSetReceived);

            captureServer.setOnReconnect([this] (int socket) {
                std::cerr << "Socket socket" << socket << " closed" << strerror(errno) << std::endl;
                sleep(1);
                captureSocket = reconnect();

                if (captureSocket == -1) {
                    throw std::runtime_error(std::format("Socket creation failed {} {}", captureSocketName, strerror(errno)));
                }

                return captureSocket;
            });
            captureServer.setSocket(reconnect());
            captureThread = std::thread([this]() {captureServer.runClient();});
        }

        void stop() {
            captureServer.stop();
            if (captureThread.joinable()) {
                captureThread.join();
            }

            if (captureSocket > 0) {
                shutdown(captureSocket, SHUT_RDWR);
                close(captureSocket);
            }
        }

        bool isRunning() {
            return captureServer.isRunning();
        }

        std::vector<cv::Mat> getNewFrames(std::vector<CaptureInfo> *captureInfo = nullptr) {
            if (!hasNewFrames) {
                std::unique_lock lock(newFramesMutex);
                newFramesAvailable.wait_for(lock, std::chrono::milliseconds(200));

                if (!hasNewFrames) {
                    return {};
                }
            }

            std::vector<cv::Mat> frames(readingFrames.size());
            std::unique_lock lock(readerLock);
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

        ~SocketCapture() {
            stop();
        };
    };
};


#endif //EMBEDDED_CV_SOCKETCAPTURE_H