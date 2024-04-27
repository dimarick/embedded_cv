#include <thread>
#include "SocketProxy.h"

void SocketProxy::onData(WebSocket *connection, const char *data) {
    auto n = send(socketFd, data, strlen(data), MSG_NOSIGNAL);

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

    readingThread = std::thread([](Server *_server, WebSocket *connection, int _socketFd, std::atomic<bool> *_running) {
        uint8_t buffer[256];
        while (*_running) {
            auto n = recv(_socketFd, buffer, sizeof(buffer), MSG_NOSIGNAL);

            if (n == 0) {
                perror("Unable to read socket");
                connection->send("Unable to read socket");
                connection->send(strerror(errno));
                _server->execute(std::function{[&connection, &buffer, n]() {
                    connection->close();
                    std::cerr << "closing connection" << std::endl;
                }});
            }

            _server->execute(std::function{[&connection, &buffer, n]() {
                connection->send(buffer, n);
                std::cerr << "sending data " << buffer << std::endl;
            }});
        }
    }, &this->server, connection, socketFd, &running);
}

void SocketProxy::onDisconnect(WebSocket *connection) {
    running = false;
    shutdown(socketFd, SHUT_RDWR);
    close(socketFd);
    _connections.erase(connection);
    std::cout << "Disconnected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress()) << "\n";
    readingThread.join();
}
