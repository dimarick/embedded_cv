#include <seasocks/StringUtil.h>
#include "SocketProxy.h"

void SocketProxy::onData(WebSocket *connection, const char *data) {
    connectionHandlers.find(connection)->second->onData(data);
}

void SocketProxy::onConnect(WebSocket *connection) {
    sockaddr addr = {AF_UNIX};
    socketName.copy(addr.sa_data, sizeof(addr.sa_data), 0);
    auto socketFd = socket(AF_UNIX, SOCK_STREAM, 0);
    auto status = connect(socketFd, &addr, sizeof(addr));

    if (status < 0) {
        perror("Unable to connect to socket");
        connection->send("Unable to connect to socket");
        connection->send(strerror(errno));
        connection->close();

        return;
    }

    auto pHandler = new ConnectionHandler(connection, socketFd);
    connectionHandlers.insert({connection, pHandler});

    pHandler->start();

    std::cout << "Connected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress())
              << "\nCredentials: " << *(connection->credentials()) << "\n";

}

void SocketProxy::onDisconnect(WebSocket *connection) {
    const auto &iterator = connectionHandlers.find(connection);
    if (iterator == connectionHandlers.end()) {
        return;
    }

    auto handler = iterator->second;
    handler->stop();
    connectionHandlers.erase(connection);
    delete handler;

    std::cout << "Disconnected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress()) << "\n";
}
