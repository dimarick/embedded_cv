#include "ConnectionHandler.h"
#include <unistd.h>

void ConnectionHandler::start() {
    readingThread = std::thread([](WebSocket *c, int _socketFd, std::atomic<bool> *_running) {
        uint8_t buffer[256];
        auto &_server = c->server();
        while (*_running) {
            auto n = recv(_socketFd, buffer, sizeof(buffer), MSG_NOSIGNAL);

            if (n == 0) {
                perror("Unable to read socket");
                c->send("Unable to read socket");
                c->send(strerror(errno));
                _server.execute(std::function{[&c, &buffer, n]() {
                    c->close();
                    std::cerr << "closing connection" << std::endl;
                }});
            }

            _server.execute(std::function{[&c, &buffer, n]() {
                c->send(buffer, n);
                std::cerr << "sending data " << buffer << std::endl;
            }});
        }
    }, connection, socketFd, &running);

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
        readingThread.join();
        shutdown(socketFd, SHUT_RDWR);
        close(socketFd);
    }
}
