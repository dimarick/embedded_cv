#include <seasocks/StringUtil.h>
#include <sys/un.h>
#include <format>
#include "SocketProxy.h"

void SocketProxy::onData(WebSocket *connection, const uint8_t *data, size_t dataSize) {
    connectionHandlers.find(connection)->second->onData(data, dataSize);
}

void SocketProxy::onData(WebSocket *connection, const char *data) {
    connectionHandlers.find(connection)->second->onData((const uint8_t *)data, strlen(data));
}

void SocketProxy::onConnect(WebSocket *connection) {
    sockaddr_un addr = {AF_UNIX};

    socketName.copy(addr.sun_path, sizeof(addr.sun_path), 0);
    auto socketFd = socket(AF_UNIX, SOCK_STREAM, 0);
    auto status = connect(socketFd, (const struct sockaddr *)&addr, sizeof(addr));

    if (status < 0) {
        perror("Unable to connect to socket");
        connection->send("Unable to connect to socket");
        connection->send(strerror(errno));
        connection->close();

        return;
    }

    int w = 0, h = 0;

    auto pHandler = new ConnectionHandler(connection, socketFd, w, h, sendingQueueDepth);
    connectionHandlers.insert({connection, pHandler});

    pHandler->start();

    std::cout << "Connected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress())
              << "\nCredentials: " << *(connection->credentials()) << "\n";

}

std::unordered_map<std::string, std::string> SocketProxy::parseQuery(const WebSocket *connection) const {
    std::unordered_map<std::string, std::string> queryParams;
    const auto &requestUri = connection->getRequestUri();
    const auto parts1 = split(requestUri, '?');
    if (parts1.size() > 1) {
        const auto parts2 = split(parts1[1], '&');
        for (const auto &param: parts2) {
            const auto keyValue = split(param, '=');
            const auto &key = keyValue[0];
            const auto &value = keyValue[1];
            queryParams[key] = value;
        }
    }

    return queryParams;
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
    connection->close();
}
