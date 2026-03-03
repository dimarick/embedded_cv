#include "RemoteView.h"
#include "SocketFactory.h"
#include "Encapsulation.h"

using namespace ecv;

std::string urlEncode(const std::string &value);

void RemoteView::showMat(const std::string& viewName, const cv::Mat& mat) {
#ifdef HAVE_OPENCV_HIGHGUI
    cv::imshow(viewName, mat);
#endif
    initializeServer();
    size_t size = 0;

    std::unordered_map<int, std::unordered_map<int, ChannelSettings>> viewSettingsCopy;

    {
        std::lock_guard lock(channelsMutex);
        const auto &viewSettings = channelSettings.find(viewName);

        if (viewSettings == channelSettings.end()) {
            return;
        }

        viewSettingsCopy = viewSettings->second;
    }
    std::unordered_map<std::string, std::vector<char>> settingsCache;

    for (const auto &socketSettings : viewSettingsCopy) {
        int socket = socketSettings.first;
        for (const auto &settings : socketSettings.second) {
            const auto &s = settings.second;
            const auto &key = std::format("{} {} {} {} {}", (int)s.x, (int)s.y, (int)s.w, (int)s.h, s.scale);
            if (settingsCache.find(key) == settingsCache.end()) {
                settingsCache[key] = createMessageFromMat(mat, cv::Rect(s.x, s.y, s.w, s.h), s.scale);
            }
            const auto &f = settingsCache[key];

            server->sendFrame(socket, f);
        }
    }
}

int RemoteView::waitKey() {
#ifdef HAVE_OPENCV_HIGHGUI
    return cv::waitKey(1);
#else
    return 0;
#endif
}

void RemoteView::initializeServer() {
    std::lock_guard lock(channelsMutex);

    if (server == nullptr) {
        server = std::shared_ptr<mini_server::BroadcastingServer>(new mini_server::BroadcastingServer);
        server->setOnMessage([this](int socket, const std::string &message) -> void {
            std::istringstream command(message);
            std::string commandName;
            command >> commandName;

            if (commandName == "CHANNEL") {
                ChannelSettings commandData;
                command >> commandData.viewName;
                command >> commandData.channelId;
                command >> commandData.x;
                command >> commandData.y;
                command >> commandData.w;
                command >> commandData.h;
                command >> commandData.scale;

                std::lock_guard lock(channelsMutex);
                const auto &viewSettings = channelSettings.find(commandData.viewName);
                if (viewSettings == channelSettings.end()) {
                    channelSettings.insert({commandData.viewName, {{socket, {{commandData.channelId, commandData}}}}});
                }
                const auto &socketSettings = viewSettings->second.find(socket);
                if (socketSettings == viewSettings->second.end()) {
                    viewSettings->second.insert({socket, {{commandData.channelId, commandData}}});
                }
                socketSettings->second.insert_or_assign(commandData.channelId, commandData);
            } else if (commandName == "DESTROY_CHANNEL") {
                ChannelSettings commandData;
                command >> commandData.viewName;
                command >> commandData.channelId;

                std::lock_guard lock(channelsMutex);
                const auto &viewSettings = channelSettings.find(commandData.viewName);
                if (viewSettings == channelSettings.end()) {
                    return;
                }
                const auto &socketSettings = viewSettings->second.find(socket);
                if (socketSettings == viewSettings->second.end()) {
                    return;
                }

                const auto &settings = socketSettings->second.find(commandData.channelId);
                if (settings == socketSettings->second.end()) {
                    return;
                }

                socketSettings->second.erase(settings);
            }
        });

        server->setOnClose([this](int socket) -> void {
            std::lock_guard lock(channelsMutex);
            for (auto &viewSettings : channelSettings) {
                viewSettings.second.erase(socket);
            }
        });

        server->setSocket(mini_server::SocketFactory::createListeningSocket("/tmp/stream/", 10));

        serverThread = std::thread([this]() {
            server->run();
            std::lock_guard lock(channelsMutex);
            channelSettings.clear();
        });
    }
}

std::vector<char> RemoteView::createMessageFromMat(const cv::Mat &mat, const cv::Rect &rect, float scale) {
    CvMatHeader header{};
    header.channels = mat.channels();
    header.x = (short)rect.x;
    header.y = (short)rect.y;
    header.w = mat.cols;
    header.h = mat.rows;
    header.scale = scale;
    header.codec = CvMatCodecEnum::RAW;

    switch (mat.type() & CV_MAT_DEPTH_MASK) {
        case CV_8U:
            header.type = CvMatTypeEnum::TYPE_8U;
            break;
        case CV_8S:
            header.type = CvMatTypeEnum::TYPE_8S;
            break;
        case CV_16S:
            header.type = CvMatTypeEnum::TYPE_16S;
            break;
        case CV_16U:
            header.type = CvMatTypeEnum::TYPE_16U;
            break;
        case CV_16BF:
            header.type = CvMatTypeEnum::TYPE_16F;
            break;
        case CV_32F:
            header.type = CvMatTypeEnum::TYPE_32F;
            break;
        case CV_64F:
            header.type = CvMatTypeEnum::TYPE_64F;
            break;
        default:
            throw std::runtime_error(std::format("Unsupported mat type {}", mat.type()));
    }

    cv::Mat crop = mat(rect).clone();
    cv::resize(crop, crop, cv::Size((int)std::round((float)crop.cols * scale), (int)std::round((float)crop.rows * scale)));

    std::vector<char> frame;

    if (header.type == TYPE_8U || header.type == TYPE_8S) {
        std::vector<uchar> buffer;
        cv::imencode("jpg", crop, buffer, {
            cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 80,
        });

        header.codec = CvMatCodecEnum::JPEG;
        return Encapsulation::encapsulate(buffer.data(), buffer.size(), header);
    }
    header.codec = CvMatCodecEnum::RAW;
    return Encapsulation::encapsulate(crop.datastart, crop.dataend - crop.datastart, header);
}

RemoteView::~RemoteView() {
    server->stop();
    if (serverThread.joinable()) {
        serverThread.join();
    }
}
