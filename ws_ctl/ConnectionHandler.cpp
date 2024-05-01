#include "ConnectionHandler.h"
#include <unistd.h>
#include <seasocks/StringUtil.h>

void ConnectionHandler::start() {
    readingThread = std::thread([](WebSocket *c, int _socketFd, ConnectionHandler *that) {
        uint8_t buffer[(size_t)1e6];
        auto &_server = c->server();
        while (true) {
            auto n = recv(_socketFd, buffer, sizeof(buffer), MSG_NOSIGNAL);

            if (!that->running) {
                break;
            }

            if (n == 0) {
                perror("Unable to read socket");
                auto err = errno;
                _server.execute(std::function{[&c, buffer, n, err]() {
                    c->send("Unable to read socket");
                    c->send(strerror(err));
                    c->close();
                    std::cerr << "closing connection" << std::endl;
                }});

                break;
            }

            _server.execute(std::function{[c, buffer, n, &that]() {
                std::cerr << "sending data " << buffer << std::endl;
                if (!that->running) {
                    return;
                }
                c->send(buffer, n);
            }});
        }

        std::cerr << "Performing graceful shutdown for " << formatAddress(c->getRemoteAddress()) << std::endl;
    }, connection, socketFd, this);

    running = true;
}

void ConnectionHandler::onData(const char *data) {
    auto n = send(socketFd, data, strlen(data), MSG_NOSIGNAL);

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
