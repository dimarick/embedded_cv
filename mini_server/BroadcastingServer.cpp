#include <iostream>
#include <string>
#include <unistd.h>
#include <netinet/in.h>
#include <cstring>
#include <stdexcept>
#include <poll.h>
#include <thread>
#include "BroadcastingServer.h"
#include "Encapsulation.h"

using namespace mini_server;

extern "C++" void BroadcastingServer::run() {
    struct pollfd fd = { .fd = socket, .events = POLLIN };
    int threadId = 0;

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

        threadId++;

        threads.insert({threadId, std::thread(BroadcastingServer::interact, acceptedSocket, this, threadId)});

        std::cout << "socket accepted: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;
    }

    std::cerr << "Performing graceful shutdown of BroadcastingServer" << std::endl;
}

void BroadcastingServer::interact(int socket, BroadcastingServer *server, int threadId) {
    char input[BUFFER_SIZE];
    struct pollfd fd = { .fd = socket, .events = POLLIN };

    while (server->running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        bzero(input,sizeof(input));
        auto n = read(socket, input, sizeof(input));

        if (errno == EPIPE) {
            break;
        }

        if (n == BUFFER_SIZE) {
            throw std::runtime_error(strerror(errno));
        }

        if (n == 0) {
            break;
        }

        if (n < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (server->onMessage != nullptr) {
            server->onMessage(socket, input);
        }
    }
}

extern "C++" void BroadcastingServer::broadcast(const std::string &message) {
    this->broadcast(message.c_str(), message.size());
}

extern "C++" void BroadcastingServer::broadcast(const void *buffer, size_t bufferSize, unsigned long ttl, MessageTypeEnum type) {
    acceptedSocketsMutex.lock();
    auto threadSafeAcceptedSockets = acceptedSockets;
    acceptedSocketsMutex.unlock();

    MessageHeader header{'MsgS', MessageTypeEnum::TYPE_MAT, (unsigned int) bufferSize, ttl};
    auto frame = Encapsulation::encapsulate(buffer, bufferSize, header);

    for (auto acceptedSocket : threadSafeAcceptedSockets) {
        sendFrame(acceptedSocket, frame);
    }
}

void BroadcastingServer::sendFrame(int s, const std::vector<char> &frame) {
    size_t sent = 0;
    while (true) {
        auto n = send(s, &frame[sent], frame.size() - sent, MSG_NOSIGNAL);

        if (n < 0) {
            perror("ERROR writing to socket");

            if (onClose != nullptr) {
                onClose(socket);
            }

            acceptedSocketsMutex.lock();
            acceptedSockets.erase(s);
            close(s);
            acceptedSocketsMutex.unlock();

            std::cerr << "socket released: " << s << ". Connections count: " << acceptedSockets.size()
                      << std::endl;

            break;
        }

        sent += n;

        if (sent == frame.size()) {
            break;
        }
    }
}

void BroadcastingServer::setOnMessage(BroadcastingServer::MessageHandler handler) {
    onMessage = std::move(handler);
}

void BroadcastingServer::setOnClose(BroadcastingServer::CloseHandler handler) {
    onClose = std::move(handler);
}
