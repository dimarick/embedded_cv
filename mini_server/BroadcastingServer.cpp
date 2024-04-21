#include <iostream>
#include <string>
#include <unistd.h>
#include <netinet/in.h>
#include <cstring>
#include <stdexcept>
#include "BroadcastingServer.h"

using namespace mini_server;

extern "C++" void BroadcastingServer::run() {
    while (true) {
        int acceptedSocket = accept(socket, nullptr, nullptr);

        if (acceptedSocket < 0) {
            throw std::runtime_error(strerror(errno));
        }

        acceptedSocketsMutex.lock();
        acceptedSockets.insert(acceptedSocket);
        acceptedSocketsMutex.unlock();
        std::cout << "socket accepted: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;
    }
}

extern "C++" void BroadcastingServer::broadcast(const std::string &message) {
    acceptedSocketsMutex.lock();
    auto threadSafeAcceptedSockets = acceptedSockets;
    acceptedSocketsMutex.unlock();

    for (auto acceptedSocket : threadSafeAcceptedSockets) {
        auto n = write(acceptedSocket, message.c_str(), message.size());

        if (n < 0) {
            perror("ERROR writing to socket");

            acceptedSocketsMutex.lock();
            acceptedSockets.erase(acceptedSocket);
            acceptedSocketsMutex.unlock();

            std::cout << "socket released: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;
        }
    }
}
