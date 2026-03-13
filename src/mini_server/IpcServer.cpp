#include <iostream>
#include <string>
#include <unistd.h>
#include <netinet/in.h>
#include <cstring>
#include <stdexcept>
#include <poll.h>
#include <thread>
#include "IpcServer.h"
#include "Encapsulation.h"
#include "common/Defer.h"
#include "common/Telemetry.h"

using namespace mini_server;

extern "C++" void IpcServer::serve() {
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

        {
            std::lock_guard lock(mutex);
            acceptedSockets.insert(acceptedSocket);

            threadId++;

            threads.insert({threadId, std::thread(IpcServer::interactThread, acceptedSocket, this, threadId)});
        }

        std::cout << "socket accepted: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;

        {
            std::lock_guard lock(mutex);
            auto _deadThreads = deadThreads;
            for (auto tid : _deadThreads) {
                const auto &it = threads.find(tid);
                if (it == threads.end()) {
                    continue;
                }

                it->second.join();
                deadThreads.erase(tid);
                threads.erase(tid);

                std::cout << tid << " thread stopped" << std::endl;
            }
        }
    }

    std::cerr << "Performing graceful shutdown of IpcServer" << std::endl;
}

void IpcServer::runClient() {
    running = true;
    interact(socket);
}

void IpcServer::interactThread(int interactionSocket, IpcServer *server, int threadId) {
    ecv::Defer removeThread([&server, threadId, interactionSocket]() {
        std::lock_guard lock(server->mutex);
        server->deadThreads.insert(threadId);
        server->acceptedSockets.erase(interactionSocket);
        close(interactionSocket);
    });

    try {
        server->interact(interactionSocket);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

void IpcServer::interact(int interactionSocket) const {
    struct pollfd fd = { .fd = interactionSocket, .events = POLLIN };

    size_t receivedSize = 0;
    size_t currentBufferSize = sizeof (MessageHeader) + 100;
    auto buffer = std::vector<char>(currentBufferSize);

    while (running) {
        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        auto n = recv(interactionSocket, buffer.data() + receivedSize, currentBufferSize - receivedSize, 0);

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

        auto header = (MessageHeader *)buffer.data();

        if (receivedSize >= sizeof (MessageHeader)) {
            // невалидное сообщение, отбрасываем весь остаток полученных данных
            if (header->magick != 'MsgS' || header->size > 100 * 1024 * 1024) {
                receivedSize = 0;
            // одно или несколько сообщений в буфере
            } else if (receivedSize >= header->size + sizeof (MessageHeader)) {
                size_t offset = 0;
                while (true) {
                    if (offset + sizeof (MessageHeader) > receivedSize) {
                        break;
                    }
                    auto h = (MessageHeader *)(&buffer[offset]);
                    auto d = &buffer[sizeof (MessageHeader) + offset];
                    auto frameSize = h->size + sizeof(MessageHeader);
                    if (offset + frameSize > receivedSize) {
                        break;
                    }
                    offset += frameSize;
                    if (onStringMessage != nullptr) {
                        const auto &message = std::string(reinterpret_cast<const char *>(d), h->size);
                        onStringMessage(interactionSocket, message);
                    }
                    if (onBinaryMessage != nullptr) {
                        onBinaryMessage(interactionSocket, d, h->size);
                    }
                };
                if (offset > 0 && receivedSize - offset > 0) {
                    memmove(buffer.data(), buffer.data() + offset, receivedSize - offset);
                }
                receivedSize -= offset;
            // единственное сообщение не поместилось в буфер: реаллоцируем и продолжаем читать
            } else if (header->size + sizeof (MessageHeader) > currentBufferSize) {
                currentBufferSize = header->size + sizeof (MessageHeader);
                buffer.resize(currentBufferSize);
            }
        }
    }
}

extern "C++" void IpcServer::broadcast(const std::string &message) {
    this->broadcast(message.c_str(), message.size());
}

extern "C++" void IpcServer::broadcast(const void *buffer, size_t bufferSize, unsigned long ttl, MessageTypeEnum type) {
    const auto expire = getExpire(ttl);
    mutex.lock();
    std::vector<int> threadSafeAcceptedSockets(acceptedSockets.size());
    std::copy(acceptedSockets.begin(), acceptedSockets.end(), threadSafeAcceptedSockets.begin());
    mutex.unlock();

    auto frame = createFrame(buffer, bufferSize, expire, type);
#pragma omp parallel for default(none) shared(frame, threadSafeAcceptedSockets)
    for (auto acceptedSocket : threadSafeAcceptedSockets) {
        sendFrame(acceptedSocket, frame);
    }
}

std::vector<char> IpcServer::createFrame(const void *buffer, size_t bufferSize, unsigned long expire,
                                         const IpcServer::MessageTypeEnum &type) const {
    MessageHeader header{'MsgS', type, (unsigned int) bufferSize, expire};

    auto frame = Encapsulation::encapsulate(buffer, bufferSize, header);
    return frame;
}

unsigned long IpcServer::getExpire(unsigned long ttl) const {
    if (ttl > 0) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count() + ttl;
    } else {
        return 0;
    }
}

void IpcServer::sendFrame(int s, const std::vector<char> &frame) {
    size_t sent = 0;
    while (true) {
        auto n = ::send(s, &frame[sent], frame.size() - sent, MSG_NOSIGNAL);

        if (n < 0) {
            perror("ERROR writing to socket");

            if (onClose != nullptr) {
                onClose(socket);
            }

            mutex.lock();
            acceptedSockets.erase(s);
            close(s);
            mutex.unlock();

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

void IpcServer::setOnMessage(StringMessageHandler handler) {
    onStringMessage = std::move(handler);
}

void IpcServer::setOnMessage(BinaryMessageHandler handler) {
    onBinaryMessage = std::move(handler);
}

void IpcServer::setOnClose(CloseHandler handler) {
    onClose = std::move(handler);
}

void IpcServer::send(int s, const void *buffer, size_t bufferSize, unsigned long ttl, IpcServer::MessageTypeEnum type) {
    const auto expire = getExpire(ttl);
    sendFrame(s, createFrame(buffer, bufferSize, expire, type));
}

void IpcServer::send(int s, const std::string &message, unsigned long ttl, IpcServer::MessageTypeEnum type) {
    send(s, message.data(), message.size(), ttl, type);
}

size_t IpcServer::getClientsCount() const {
    return acceptedSockets.size();
}
