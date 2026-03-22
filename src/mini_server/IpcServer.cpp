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

            threads.insert({threadId, std::thread(interactThread, acceptedSocket, this, threadId)});
            sendingTasks[acceptedSocket] = std::make_shared<SendingTask>();
            sendingThreads.insert({threadId, std::thread(sendingThread, acceptedSocket, this, threadId)});
        }

        std::cout << "socket accepted: " << acceptedSocket << ". Connections count: " << acceptedSockets.size() << std::endl;

        std::set<int> _deadThreads;
        {
            std::lock_guard lock(mutex);
            _deadThreads = deadThreads;
        }

        for (auto tid : _deadThreads) {
            std::unordered_map<int, std::thread>::iterator it;
            std::unordered_map<int, std::thread>::iterator it2;

            {
                std::lock_guard lock(mutex);
                it = threads.find(tid);
                it2 = sendingThreads.find(tid);
            }

            if (it == threads.end()) {
                continue;
            }
            it->second.join();
            if (it2 == sendingThreads.end()) {
                continue;
            }
            it2->second.join();

            std::cout << tid << " thread stopped" << std::endl;
        }
        {
            std::lock_guard lock(mutex);

            for (auto tid : _deadThreads) {
                threads.erase(tid);
                sendingThreads.erase(tid);
                deadThreads.erase(tid);
            }
        }
    }

    std::cerr << "Performing graceful shutdown of IpcServer" << std::endl;
}

void IpcServer::runClient() {
    use_defer;
    running = true;
    defer(running = false);
    interact(socket, 0);
}

void IpcServer::sendingThread(int interactionSocket, IpcServer *server, int threadId) {
    use_defer;
    defer ({
        std::lock_guard lock(server->mutex);
        server->sendingTasks.erase(interactionSocket);
    });

    while (server->running) {
        std::shared_ptr<SendingTask> task;
        {
            std::lock_guard lock(server->mutex);
            if (server->deadThreads.contains(threadId)) {
                break;
            }

            const auto &it = server->sendingTasks.find(interactionSocket);
            if (it == server->sendingTasks.end()) {
                break;
            }
            task = it->second;
        }
        if (!task->ready) {
            std::unique_lock lock(server->sendingMutex);
            server->sendingCV.wait_for(lock, std::chrono::milliseconds(200));
            continue;
        }
        task->ready = false;
        server->sendFrame(interactionSocket, task->buffer);
        task->done = true;
    }
}

void IpcServer::interactThread(int interactionSocket, IpcServer *server, int threadId) {
    use_defer;
    defer ({
        std::lock_guard lock(server->mutex);
        server->deadThreads.insert(threadId);
        server->acceptedSockets.erase(interactionSocket);
        close(interactionSocket);
    });

    try {
        server->interact(interactionSocket, threadId);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
}

void IpcServer::interact(int interactionSocket, int threadId) const {
   struct pollfd fd = { .fd = interactionSocket, .events = POLLIN };

    size_t receivedSize = 0;
    size_t currentBufferSize = sizeof (MessageHeader) + 100;
    auto buffer = std::vector<char>(currentBufferSize);

    while (running) {
        {
            std::lock_guard lock(mutex);
            if (threadId != 0 && deadThreads.contains(threadId)) {
                break;
            }
        }

        int pollStatus = poll(&fd, 1, 50);
        if (pollStatus < 0) {
            throw std::runtime_error(strerror(errno));
        }

        if (pollStatus == 0) {
            continue;
        }

        auto n = recv(interactionSocket, buffer.data() + receivedSize, currentBufferSize - receivedSize, 0);

        if ((fd.revents & POLLHUP || errno == EPIPE || errno == ENOENT || errno == ECONNRESET) && n <= 0) {
            auto s = onReconnect(interactionSocket);
            close(interactionSocket);
            interactionSocket = s;
            fd.fd = interactionSocket;
            continue;
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
                    MessageHeader h = {};
                    memcpy(&h, &buffer[offset], sizeof(h));

                    if (sizeof (MessageHeader) + offset >= buffer.size()) {
                        break;
                    }

                    auto d = &buffer[sizeof (MessageHeader) + offset];
                    auto frameSize = h.size + sizeof(MessageHeader);
                    if (offset + frameSize > receivedSize) {
                        break;
                    }
                    offset += frameSize;
                    if (onStringMessage != nullptr) {
                        const auto message = std::string(d, h.size);
                        onStringMessage(interactionSocket, message);
                    }
                    if (onBinaryMessage != nullptr) {
                        onBinaryMessage(interactionSocket, d, h.size);
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
    auto threadSafeSendingTasks = sendingTasks;
    std::copy(acceptedSockets.begin(), acceptedSockets.end(), threadSafeAcceptedSockets.begin());
    mutex.unlock();

    auto frame = createFrame(buffer, bufferSize, expire, type);

    std::atomic hasReady = false;

    for (int acceptedSocket : threadSafeAcceptedSockets) {
        auto t = threadSafeSendingTasks[acceptedSocket];
        if (t->done == false) {
            continue;
        }

        t->done = false;
        t->buffer = frame;
        t->ready = true;
        hasReady = true;
    }

    if (hasReady) {
        sendingCV.notify_all();
    }
}

std::vector<char> IpcServer::createFrame(const void *buffer, size_t bufferSize, unsigned long expire,
                                         const IpcServer::MessageTypeEnum &type) const {
    MessageHeader header{'MsgS', type, (unsigned int) bufferSize, expire};

    auto frame = Encapsulation::encapsulate(buffer, bufferSize, header);
    return frame;
}

unsigned long IpcServer::getExpire(unsigned long ttl, long frameCreatedAt) const {
    if (ttl > 0) {
        auto now = frameCreatedAt == 0
            ? std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()
            : frameCreatedAt;

        return now + ttl * 1000;
    }

    return 0;
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

void IpcServer::setOnReconnect(ReconnectHandler handler) {
    onReconnect = std::move(handler);
}

void IpcServer::send(int s, const void *buffer, size_t bufferSize, unsigned long ttl, MessageTypeEnum type) {
    const auto expire = getExpire(ttl);
    sendFrame(s, createFrame(buffer, bufferSize, expire, type));
}

void IpcServer::send(int s, const char *cString, unsigned long ttl, MessageTypeEnum type) {
    const auto expire = getExpire(ttl);
    sendFrame(s, createFrame(cString, strlen(cString), expire, type));
}

void IpcServer::send(int s, const std::string &message, unsigned long ttl, MessageTypeEnum type) {
    send(s, message.data(), message.size(), ttl, type);
}

size_t IpcServer::getClientsCount() const {
    return acceptedSockets.size();
}
