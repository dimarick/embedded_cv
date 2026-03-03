#include "RemoteView.h"
#include "SocketFactory.h"

using namespace ecv;

std::string urlEncode(const std::string &value);

void RemoteView::showMat(const std::string& viewName, const cv::Mat& mat) {
#ifdef HAVE_OPENCV_HIGHGUI
    cv::imshow(viewName, mat);
#endif
    auto &channel = getOrCreateChannel(viewName);
    size_t size = 0;
    const auto message = createMessageFromMat(mat, &size);
    channel.server->broadcast(message.get(), size);
}

int RemoteView::waitKey() {
#ifdef HAVE_OPENCV_HIGHGUI
    return cv::waitKey(1);
#else
    return 0;
#endif
}

const RemoteView::Channel &RemoteView::getOrCreateChannel(const std::string &name) {
    std::lock_guard lock(channelsMutex);
    const auto &channelIt = channels.find(name);
    if (channelIt != channels.end()) {
        return channelIt->second;
    }

    auto serverRef = std::shared_ptr<mini_server::BroadcastingServer>(new mini_server::BroadcastingServer);

    serverRef->setSocket(mini_server::SocketFactory::createListeningSocket(std::string("/tmp/stream/") + urlEncode(name), 10));

    return channels.insert({name, Channel{
        name,
        serverRef,
        std::thread([&serverRef]() {
            serverRef->run();
        })
    }}).first->second;
}

std::shared_ptr<RemoteView::CvMatHeader> RemoteView::createMessageFromMat(const cv::Mat &mat, size_t *size) {
    size_t headerSize = sizeof(CvMatHeader);
    size_t dataSize = mat.dataend - mat.datastart;
    auto bufferSize = headerSize + dataSize;

    char *buffer = new char[bufferSize];
    auto header = reinterpret_cast<CvMatHeader *>(buffer);
    auto body = reinterpret_cast<void *>(buffer + headerSize);

    memcpy(body, mat.datastart, dataSize);

    header->channels = mat.channels();
    header->w = mat.cols;
    header->h = mat.rows;

    switch (mat.type() & CV_MAT_DEPTH_MASK) {
        case CV_8U:
            header->type = CvMatTypeEnum::TYPE_8U;
            break;
        case CV_8S:
            header->type = CvMatTypeEnum::TYPE_8S;
            break;
        case CV_16S:
            header->type = CvMatTypeEnum::TYPE_16S;
            break;
        case CV_16U:
            header->type = CvMatTypeEnum::TYPE_16U;
            break;
        case CV_16BF:
            header->type = CvMatTypeEnum::TYPE_16F;
            break;
        case CV_32F:
            header->type = CvMatTypeEnum::TYPE_32F;
            break;
        case CV_64F:
            header->type = CvMatTypeEnum::TYPE_64F;
            break;
        default:
            throw std::runtime_error(std::format("Unsupported mat type {}", mat.type()));
    }

    return std::shared_ptr<CvMatHeader>(header);
}

RemoteView::~RemoteView() {
    for (auto &channel : channels) {
        channel.second.server->stop();
    }
    for (auto &channel : channels) {
        if (channel.second.thread.joinable()) {
            channel.second.thread.join();
        }
    }
}

std::string urlEncode(const std::string &value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (const auto &c : value) {
        // Keep alphanumeric and other accepted characters intact
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
            continue;
        }

        // Any other characters are percent-encoded
        escaped << std::uppercase;
        escaped << '%' << std::setw(2) << int((unsigned char) c);
        escaped << std::nouppercase;
    }

    return escaped.str();
}
