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
    acceptedSocketsMutex.lock();
    auto threadSafeAcceptedSockets = acceptedSockets;
    acceptedSocketsMutex.unlock();

    for (auto acceptedSocket : threadSafeAcceptedSockets) {
        auto n = send(acceptedSocket, message.c_str(), message.size(), MSG_NOSIGNAL);

        if (n < 0) {
            perror("ERROR writing to socket");

            acceptedSocketsMutex.lock();
            acceptedSockets.erase(acceptedSocket);
            acceptedSocketsMutex.unlock();

            std::cerr << "socket released: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;
        }
        std::cerr << "Message : " << message << " sent to " << acceptedSocket << std::endl;
    }
}
