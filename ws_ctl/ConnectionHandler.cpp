#include "ConnectionHandler.h"
#include <unistd.h>
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
                _server.execute(std::function{[c, buffer, data, messageSize, &that]() {
                    std::cerr << "sending data " << data << std::endl;
                    if (!that->running) {
                        free((void *)buffer);
                        return;
                    }
                    c->send(data, messageSize);
                }});
                messageSize = 0;
            }
        }

        std::cerr << "Performing graceful shutdown for " << formatAddress(c->getRemoteAddress()) << std::endl;

        shutdown(_socketFd, SHUT_RDWR);
        close(_socketFd);
    }, connection, socketFd, this);

    running = true;
}

void ConnectionHandler::onData(const uint8_t *data, size_t dataSize) {
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
