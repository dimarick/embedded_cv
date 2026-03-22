#include "ConnectionHandler.h"
#include <unistd.h>
#include <semaphore>
#include <poll.h>
#include <seasocks/StringUtil.h>
#include <seasocks/Connection.h>
#include <netinet/tcp.h>

void ConnectionHandler::start() {
    readingThread = std::thread([this](WebSocket *c, int _socketFd) {
        if (auto connection = dynamic_cast<seasocks::Connection *>(c); connection != nullptr) {
            auto socket = connection->getFd();
            int setTrue = 1;
            setsockopt(socket, SOL_TCP, TCP_NODELAY, &setTrue, sizeof(setTrue));
        }
        size_t currentBufferSize = sizeof (MessageHeader) + 100;
        auto buffer = (uint8_t *) malloc(currentBufferSize);
        size_t messageSize = 0;

        auto header = (MessageHeader *)buffer;
        auto data = &buffer[sizeof (MessageHeader)];

        size_t receivedSize = 0;
        auto &_server = c->server();


        struct pollfd fd = { .fd = _socketFd, .events = POLLIN };
        while (running) {
            int pollStatus = poll(&fd, 1, 50);
            if (pollStatus < 0) {
                throw std::runtime_error(strerror(errno));
            }

            if (pollStatus == 0) {
                continue;
            }

            auto n = recv(_socketFd, buffer + receivedSize, currentBufferSize - receivedSize, 0);

            if (n <= 0) {
                perror("Unable to read socket");
                auto err = errno;
                std::mutex shudownMutex;

                shudownMutex.lock();

                _server.execute(std::function{[&c, buffer, n, err, &shudownMutex]() {
                    c->send("Unable to read socket");
                    c->send(strerror(err));
                    shudownMutex.unlock();
                    c->close();
                    std::cerr << "closing connection" << std::endl;
                }});

                // Дожидаемся завершения _server.execute
                shudownMutex.lock();
                shudownMutex.unlock();
                selfStop();

                break;
            }

            receivedSize += n;

            if (receivedSize >= sizeof (MessageHeader)) {
                if (header->magick != 'MsgS' || header->size > 100 * 1024 * 1024) {
                    messageSize = 0;
                    receivedSize = 0;
                } else if (receivedSize >= header->size + sizeof (MessageHeader)) {
                    messageSize = header->size;
                    receivedSize -= header->size + sizeof (MessageHeader);

                    if (receivedSize > 0) {
                        memmove(buffer, buffer + header->size + sizeof (MessageHeader), currentBufferSize - (header->size + sizeof (MessageHeader)));
                    }
                } else if (header->size + sizeof (MessageHeader) > currentBufferSize) {
                    buffer = (uint8_t *) realloc(buffer, header->size + sizeof (MessageHeader));
                    header = (MessageHeader *)buffer;
                    data = &buffer[sizeof (MessageHeader)];
                    currentBufferSize = header->size + sizeof (MessageHeader);
                }
            }

            if (messageSize > 0) {
                if (sendingQueueDepth != -1) {
                    auto now = duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

                    if (!(header->ttl == 0 || header->ttl > now)) {
                        std::cerr << "Dropped2 frame size " << messageSize << " ttl expired " << now - header->ttl << std::endl;
                        continue;
                    }
                    std::chrono::time_point<std::chrono::system_clock> atime(std::chrono::microseconds(header->ttl));
                    auto acquired = sendingMutex.try_acquire_until(atime);

                    if (!acquired) {
                        std::cerr << "Warning try_acquire_until timed out" << std::endl;
                    }

                    now = duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    if (header->ttl == 0 || header->ttl > now) {

                        std::cerr << "Sending frame size " << messageSize << " ttl remaining " << header->ttl - now << " us" << std::endl;

                        const uint8_t *copyOfData = (uint8_t *)malloc(messageSize);
                        memcpy((void *)copyOfData, data, messageSize);

                        _server.execute(std::function{[c, copyOfData, messageSize, this]() {
                            if (!running) {
                                sendingMutex.release();

                                return;
                            }

                            c->send(copyOfData, messageSize);
                            sendingMutex.release();
                            free((void *)copyOfData);
                        }});
                    } else {
                        std::cerr << "Dropped frame size " << messageSize << " ttl expired " << now - header->ttl << std::endl;
                    }
                } else {
                    _server.execute(std::function{[c, data, messageSize, this]() {
                        if (!running) {
                            return;
                        }

                        c->send(data, messageSize);
                    }});
                }

                messageSize = 0;
            }
            usleep(100);
        }

        std::cerr << "Performing graceful shutdown for " << formatAddress(c->getRemoteAddress()) << std::endl;

        shutdown(_socketFd, SHUT_RDWR);
        close(_socketFd);
        free((void *) buffer);
    }, connection, socketFd);

    running = true;
}

void ConnectionHandler::onData(const uint8_t *data, size_t dataSize) {
    auto *frame = new uint8_t[dataSize + sizeof(MessageHeader)];
    auto *header = reinterpret_cast<MessageHeader *>(frame);
    auto *body = (frame + sizeof(MessageHeader));
    header->magick = 'MsgS';
    header->ttl = 0;
    header->type = TYPE_CONTROL;
    header->size = dataSize;
    memcpy(body, data, dataSize);
    auto n = send(socketFd, frame, dataSize + sizeof(MessageHeader), MSG_NOSIGNAL);

    delete[] frame;

    if (n == 0) {
        perror("Unable to write to socket");
        connection->send("Unable to write to socket");
        connection->send(strerror(errno));
        connection->close();
    }
}

void ConnectionHandler::stop() {
    std::cerr << "Stopping connectionHandler" << std::endl;
    if (running) {
        running = false;
        if (readingThread.joinable()) {
            readingThread.join();
        }
        std::cerr << "Stopped connectionHandler" << std::endl;
        shutdown(socketFd, SHUT_RDWR);
        close(socketFd);
    } else {
        if (readingThread.joinable()) {
            readingThread.join();
        }
    }
}

void ConnectionHandler::selfStop() {
    if (running) {
        running = false;
        shutdown(socketFd, SHUT_RDWR);
        close(socketFd);
    }
}
