#include "RemoteView.h"

#include <ranges>

#include "SocketFactory.h"
#include "Encapsulation.h"

using namespace ecv;

void RemoteView::showMat(const std::string& viewName, const cv::UMat& mat) {
    showMat(viewName, mat.getMat(cv::ACCESS_READ));
}

void RemoteView::showMat(const std::string& viewName, const cv::Mat& mat) {
    initializeServer();
    const auto expire = server->getExpire(200);
#ifdef HAVE_OPENCV_HIGHGUI
    cv::imshow(viewName, mat);
#endif

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
            const auto &key = std::format("{} {} {} {} {} {}", (int)s.x, (int)s.y, (int)s.w, (int)s.h, (int)s.viewW, (int)s.viewH);
            if (!settingsCache.contains(key)) {
                StringHeader str{};
                str.nameSize = (short)viewName.size();
                auto frame = mini_server::Encapsulation::encapsulate(viewName.data(), viewName.size(), str);
                auto header = createMessageHeaderFromMat(mat, cv::Rect(s.x, s.y, s.w, s.h), s.viewW, s.viewH);
                auto matMessage = createMessageFromMat(header, mat, cv::Rect(s.x, s.y, s.w, s.h), s.viewW, s.viewH);

                auto nameBufferSize = frame.size();
                frame.resize(nameBufferSize + matMessage.size());

                std::copy(matMessage.begin(), matMessage.end(), frame.begin() + (int)nameBufferSize);

                settingsCache[key] = server->createFrame(frame.data(), frame.size(), expire,
                                                         mini_server::IpcServer::MessageTypeEnum::TYPE_MAT);

                {
                    std::shared_lock lock(viewsMutex);
                    views.insert_or_assign(viewName, header);
                }
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
        server = std::make_shared<mini_server::IpcServer>();
        server->setOnMessage([this](int socket, const std::string &message) -> void {
            std::cout << message << std::endl;

            std::istringstream command(message);
            std::string commandName;
            command >> commandName;

            if (commandName == "CHANNEL") {
                ChannelSettings commandData;
                command >> std::quoted(commandData.viewName);
                command >> commandData.channelId;
                command >> commandData.viewW;
                command >> commandData.viewH;
                command >> commandData.x;
                command >> commandData.y;
                command >> commandData.w;
                command >> commandData.h;

                if (commandData.channelId < 0) {
                    std::cerr << "Failed to parse data" << std::endl;
                    return;
                }

                std::lock_guard lock(channelsMutex);
                channelSettings.insert({commandData.viewName, {{socket, {{commandData.channelId, commandData}}}}});
                const auto &viewSettings = channelSettings.find(commandData.viewName);
                viewSettings->second.insert({socket, {{commandData.channelId, commandData}}});
                const auto &socketSettings = viewSettings->second.find(socket);
                socketSettings->second.insert_or_assign(commandData.channelId, commandData);
            } else if (commandName == "DESTROY_CHANNEL") {
                ChannelSettings commandData;
                command >> std::quoted(commandData.viewName);
                command >> commandData.channelId;

                std::lock_guard lock(channelsMutex);
                const auto &viewSettings = channelSettings.find(commandData.viewName);
                if (viewSettings == channelSettings.end()) {
                    std::cerr << "Channel failed unsubscribe 1 " << commandData.viewName << " " << socket << " " << commandData.channelId << std::endl;
                    return;
                }
                const auto &socketSettings = viewSettings->second.find(socket);
                if (socketSettings == viewSettings->second.end()) {
                    std::cerr << "Channel failed unsubscribe 2 " << commandData.viewName << " " << socket << " " << commandData.channelId << std::endl;
                    return;
                }

                const auto &settings = socketSettings->second.find(commandData.channelId);
                if (settings == socketSettings->second.end()) {
                    std::cerr << "Channel failed unsubscribe 3 " << commandData.viewName << " " << socket << " " << commandData.channelId << std::endl;
                    return;
                }

                socketSettings->second.erase(settings);

                std::cerr << "Channel unsubscribed " << commandData.viewName << " " << socket << " " << commandData.channelId << std::endl;
            }
        });

        server->setOnClose([this](int socket) -> void {
            std::lock_guard lock(channelsMutex);
            for (auto &viewSettings: channelSettings | std::views::values) {
                viewSettings.erase(socket);
            }
        });

        server->setSocket(mini_server::SocketFactory::createServerSocket(socketPath, 10));

        serverThread = std::thread([this]() {
            server->serve();
            std::lock_guard lock(channelsMutex);
            channelSettings.clear();
        });
    }
}

std::vector<char> RemoteView::createMessageFromMat(CvMatHeader header, const cv::Mat &mat, const cv::Rect &rect, short viewW, short viewH) {
    cv::Mat crop;

    if (rect.width > 0 && rect.height > 0) {
        crop = mat(rect).clone();
    } else {
        crop = mat;
    }

    if (viewW != 0 || viewH != 0) { // autofit
        auto scaleW = viewW > 0 ? (double)viewW / (double)crop.cols : 1.;
        auto scaleH = viewH > 0 ? (double)viewH / (double)crop.rows : 1.;
        auto scale = std::min(1., std::min(scaleW, scaleH));

        cv::resize(crop, crop, cv::Size(0, 0), scale, scale);
    }

    std::vector<char> frame;

    if (header.codec == JPEG) {
        std::vector<uchar> buffer;
        cv::imencode(".jpg", crop, buffer, {
            cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 50,
        });
        return mini_server::Encapsulation::encapsulate(buffer.data(), buffer.size(), header);
    }
    return mini_server::Encapsulation::encapsulate(crop.datastart, crop.dataend - crop.datastart, header);
}

RemoteView::CvMatHeader RemoteView::createMessageHeaderFromMat(const cv::Mat &mat, const cv::Rect &rect, short viewW, short viewH) {
    CvMatHeader header{};
    header.channels = (short)mat.channels();
    header.viewW = (short)(viewW & 0xFFFF);
    header.viewH = (short)(viewH & 0xFFFF);
    header.x = (short)(rect.x & 0xFFFF);
    header.y = (short)(rect.y & 0xFFFF);
    header.w = (short)(rect.width & 0xFFFF);
    header.h = (short)(rect.height & 0xFFFF);
    header.codec = RAW;

    switch (mat.type() & CV_MAT_DEPTH_MASK) {
        case CV_8U:
            header.type = TYPE_8U;
            break;
        case CV_8S:
            header.type = TYPE_8S;
            break;
        case CV_16S:
            header.type = TYPE_16S;
            break;
        case CV_16U:
            header.type = TYPE_16U;
            break;
        case CV_16BF:
            header.type = TYPE_16F;
            break;
        case CV_32F:
            header.type = TYPE_32F;
            break;
        case CV_64F:
            header.type = TYPE_64F;
            break;
        default:
            throw std::runtime_error(std::format("Unsupported mat type {}", mat.type()));
    }

    cv::Mat crop;

    if (rect.width > 0 && rect.height > 0) {
        crop = mat(rect).clone();
    } else {
        crop = mat;
    }

    if (viewW != 0 || viewH != 0) { // autofit
        auto scaleW = viewW > 0 ? (double)viewW / (double)crop.cols : 1.;
        auto scaleH = viewH > 0 ? (double)viewH / (double)crop.rows : 1.;
        auto scale = std::min(1., std::min(scaleW, scaleH));

        cv::resize(crop, crop, cv::Size(0, 0), scale, scale);
    }

    std::vector<char> frame;

    if (header.type == TYPE_8U || header.type == TYPE_8S) {
        header.codec = JPEG;
        return header;
    }
    header.codec = RAW;
    return header;
}

RemoteView::~RemoteView() {
    server->stop();
    if (serverThread.joinable()) {
        serverThread.join();
    }
}
