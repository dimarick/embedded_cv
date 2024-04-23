#include <thread>
#include "SocketProxy.h"

void SocketProxy::onData(WebSocket *connection, const char *data) {
    auto n = write(socketFd, data, strlen(data));

    if (n == 0) {
        perror("Unable to write to socket");
        connection->send("Unable to write to socket");
        connection->send(strerror(errno));
        connection->close();

        return;
    }
}

void SocketProxy::onConnect(WebSocket *connection) {
    sockaddr addr = {AF_UNIX};
    socketName.copy(addr.sa_data, sizeof(addr.sa_data), 0);
    socketFd = socket(AF_UNIX, SOCK_STREAM, 0);
    auto status = connect(socketFd, &addr, sizeof(addr));

    if (status < 0) {
        perror("Unable to connect to socket");
        connection->send("Unable to connect to socket");
        connection->send(strerror(errno));
        connection->close();

        return;
    }

    _connections.insert(connection);

    std::cout << "Connected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress())
              << "\nCredentials: " << *(connection->credentials()) << "\n";

    auto pid = fork();

    if (pid == 0) {
        uint8_t buffer[256];
        while (true) {
            auto n = read(socketFd, buffer, sizeof(buffer));

            if (n == 0) {
                perror("Unable to read socket");
                break;
            }

            this->server.execute(std::function{[&connection, &buffer, n]() {
                connection->send(buffer, n);
            }});
        }

        exit(0);
    }

    readingThread = std::thread([](Server *_server, WebSocket *connection, int _socketFd, std::atomic<bool> *_running) {
        uint8_t buffer[256];
        while (*_running) {
            auto n = read(_socketFd, buffer, sizeof(buffer));

            if (n == 0) {
                perror("Unable to read socket");
                break;
            }

            _server->execute(std::function{[&connection, &buffer, n]() {
                connection->send(buffer, n);
                std::cerr << "sending data" << buffer << std::endl;
            }});
        }
    }, &this->server, connection, socketFd, &running);
}

void SocketProxy::onDisconnect(WebSocket *connection) {
    running = false;
    close(socketFd);
    _connections.erase(connection);
    std::cout << "Disconnected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress()) << "\n";
    readingThread.join();
}
