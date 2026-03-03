#include <iostream>
#include <string>
#include <unistd.h>
#include <netinet/in.h>
#include <cstring>
#include <stdexcept>
#include <poll.h>
#include "BroadcastingServer.h"

using namespace mini_server;

extern "C++" void BroadcastingServer::run() {
    struct pollfd fd = { .fd = socket, .events = POLLIN };

    running = true;

    while (running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        int acceptedSocket = accept(socket, nullptr, nullptr);

        if (acceptedSocket < 0) {
            throw std::runtime_error(strerror(errno));
        }

        acceptedSocketsMutex.lock();
        acceptedSockets.insert(acceptedSocket);
        acceptedSocketsMutex.unlock();
        std::cout << "socket accepted: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;
    }

    std::cerr << "Performing graceful shutdown of BroadcastingServer" << std::endl;
}

extern "C++" void BroadcastingServer::broadcast(const std::string &message) {
    this->broadcast(message.c_str(), message.size());
}

extern "C++" void BroadcastingServer::broadcast(const void *buffer, size_t bufferSize, unsigned long ttl, MessageTypeEnum type) {
    acceptedSocketsMutex.lock();
    auto threadSafeAcceptedSockets = acceptedSockets;
    acceptedSocketsMutex.unlock();

    size_t frameSize = bufferSize + sizeof (MessageHeader);
    const char *frame = (char *)malloc(frameSize);
    auto header = (MessageHeader *)frame;
    auto data = &frame[sizeof (MessageHeader)];
    memcpy((void *) data, buffer, bufferSize);
    header->magick = 'MsgS';
    header->type = type;
    header->size = bufferSize;
    if (ttl > 0) {
        header->ttl = duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count() + ttl;
    } else {
        header->ttl = 0;
    }

    for (auto acceptedSocket : threadSafeAcceptedSockets) {
        size_t sent = 0;
        while (true) {
            auto n = send(acceptedSocket, frame + sent, frameSize - sent, MSG_NOSIGNAL);

            if (n < 0) {
                perror("ERROR writing to socket");

                acceptedSocketsMutex.lock();
                acceptedSockets.erase(acceptedSocket);
                close(acceptedSocket);
                acceptedSocketsMutex.unlock();

                std::cerr << "socket released: " << acceptedSocket << ". Connections count: " << acceptedSockets.size()
                          << std::endl;

                break;
            }

            sent += n;

            if (sent == frameSize) {
                break;
            }
        }
    }

    free((void *)frame);
}
