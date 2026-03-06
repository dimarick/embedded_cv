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
    struct pollfd fd = { .fd = socket, .events = POLLIN };

    size_t receivedSize = 0;
    size_t currentBufferSize = sizeof (MessageHeader) + 100;
    auto buffer = (uint8_t *) malloc(currentBufferSize);
    while (server->running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        auto n = recv(socket, buffer + receivedSize, currentBufferSize - receivedSize, MSG_NOSIGNAL);

        if (errno == EPIPE) {
            break;
        }

        if (n == 0) {
            break;
        }

        if (n < 0) {
            throw std::runtime_error(strerror(errno));
        }

        receivedSize += n;

        auto header = (MessageHeader *)buffer;

        if (receivedSize >= sizeof (MessageHeader)) {
            // невалидное сообщение, отбрасываем весь остаток полученных данных
            if (header->magick != 'MsgS' || header->size > 100 * 1024 * 1024) {
                receivedSize = 0;
            // одно или несколько сообщений в буфере
            } else if (receivedSize >= header->size + sizeof (MessageHeader)) {
                size_t offset = 0;
                while (true) {
                    auto h = (MessageHeader *)(&buffer[offset]);
                    auto d = &buffer[sizeof (MessageHeader) + offset];
                    auto frameSize = h->size + sizeof(MessageHeader);
                    if (offset + frameSize > receivedSize) {
                        break;
                    }
                    offset += frameSize;
                    if (server->onMessage != nullptr) {
                        const auto &message = std::string(reinterpret_cast<const char *>(d),
                                                                             h->size);
                        server->onMessage(socket, message);
                    }
                };
                if (offset > 0 && receivedSize - offset > 0) {
                    memmove(buffer, buffer + offset, receivedSize - offset);
                }
                receivedSize -= offset;
            // единственное сообщение не поместилось в буфер: реаллоцируем и продолжаем читать
            } else if (header->size + sizeof (MessageHeader) > currentBufferSize) {
                currentBufferSize = header->size + sizeof (MessageHeader);
            }
        }
    }
}

extern "C++" void BroadcastingServer::broadcast(const std::string &message) {
    this->broadcast(message.c_str(), message.size());
}

extern "C++" void BroadcastingServer::broadcast(const void *buffer, size_t bufferSize, unsigned long ttl, MessageTypeEnum type) {
    const auto expire = getExpire(ttl);
    acceptedSocketsMutex.lock();
    auto threadSafeAcceptedSockets = acceptedSockets;
    acceptedSocketsMutex.unlock();

    auto frame = getTransportMessage(buffer, bufferSize, expire, type);

    for (auto acceptedSocket : threadSafeAcceptedSockets) {
        sendFrame(acceptedSocket, frame);
    }
}

std::vector<char> BroadcastingServer::getTransportMessage(const void *buffer, size_t bufferSize, unsigned long expire,
                                                          const BroadcastingServer::MessageTypeEnum &type) const {
    MessageHeader header{'MsgS', type, (unsigned int) bufferSize, expire};

    auto frame = Encapsulation::encapsulate(buffer, bufferSize, header);
    return frame;
}

unsigned long BroadcastingServer::getExpire(unsigned long ttl) const {
    if (ttl > 0) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count() + ttl;
    } else {
        return 0;
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
