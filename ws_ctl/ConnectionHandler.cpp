#include "ConnectionHandler.h"
#include <unistd.h>
#include <semaphore>
#include <seasocks/StringUtil.h>

void ConnectionHandler::start() {
    readingThread = std::thread([](WebSocket *c, int _socketFd, ConnectionHandler *that) {
        size_t currentBufferSize = sizeof (MessageHeader) + 100;
        auto buffer = (uint8_t *) malloc(currentBufferSize);
        size_t messageSize = 0;

        auto header = (MessageHeader *)buffer;
        auto data = &buffer[sizeof (MessageHeader)];

        size_t receivedSize = 0;
        auto &_server = c->server();
        while (that->running) {
            auto n = recv(_socketFd, buffer + receivedSize, currentBufferSize - receivedSize, MSG_NOSIGNAL);

            if (n < 0) {
                perror("Unable to read socket");
                auto err = errno;
                std::mutex shudownMutex;

                shudownMutex.lock();

                _server.execute(std::function{[&c, buffer, n, err, &shudownMutex]() {
                    c->send("Unable to read socket");
                    c->send(strerror(err));
                    c->close();
                    std::cerr << "closing connection" << std::endl;
                    shudownMutex.unlock();
                }});

                // Дожидаемся завершения _server.execute
                shudownMutex.lock();
                shudownMutex.unlock();
                that->stop();

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
                        memcpy(buffer, buffer + header->size + sizeof (MessageHeader), currentBufferSize - (header->size + sizeof (MessageHeader)));
                    }
                } else if (header->size + sizeof (MessageHeader) > currentBufferSize) {
                    buffer = (uint8_t *) realloc(buffer, header->size + sizeof (MessageHeader));
                    header = (MessageHeader *)buffer;
                    data = &buffer[sizeof (MessageHeader)];
                    currentBufferSize = header->size + sizeof (MessageHeader);
                }
            }

            if (messageSize > 0) {
                if (that->sendingQueueDepth != -1) {
                    that->sendingMutex.try_acquire_until(std::chrono::time_point<std::chrono::system_clock>(std::chrono::milliseconds(header->ttl)));

                    auto now = duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
                    if (header->ttl == 0 || header->ttl > now) {

                        std::cerr << "Sending frame size " << messageSize << " ttl remaining " << header->ttl - now << std::endl;

                        const uint8_t *copyOfData = (uint8_t *)malloc(messageSize);
                        memcpy((void *)copyOfData, data, messageSize);

                        _server.execute(std::function{[c, copyOfData, messageSize, &that]() {
                            if (!that->running) {
                                that->sendingMutex.release();

                                return;
                            }

                            c->send(copyOfData, messageSize);
                            free((void *)copyOfData);
                        }});
                    } else {
                        std::cerr << "Dropped frame size " << messageSize << " ttl expired " << now - header->ttl << std::endl;
                    }
                } else {
                    _server.execute(std::function{[c, data, messageSize, &that]() {
                        if (!that->running) {
                            return;
                        }

                        c->send(data, messageSize);
                    }});
                }

                messageSize = 0;
            }
        }

        std::cerr << "Performing graceful shutdown for " << formatAddress(c->getRemoteAddress()) << std::endl;

        shutdown(_socketFd, SHUT_RDWR);
        close(_socketFd);
        free((void *) buffer);
    }, connection, socketFd, this);

    running = true;
}

void ConnectionHandler::onData(const uint8_t *data, size_t dataSize) {
    if (this->sendingQueueDepth != -1) {
        if (dataSize == 3 && strncmp((const char *)(data), "ACK", 3) == 0) {
            std::cerr << "ACK received" << std::endl;

            sendingMutex.release();
            return;
        }
    }

    auto n = send(socketFd, data, dataSize, MSG_NOSIGNAL);

    if (n == 0) {
        perror("Unable to write to socket");
        connection->send("Unable to write to socket");
        connection->send(strerror(errno));
        connection->close();

        return;
    }
}

void ConnectionHandler::stop() {
    if (running) {
        running = false;
        shutdown(socketFd, SHUT_RDWR);
        close(socketFd);
        readingThread.join();
    }
}
