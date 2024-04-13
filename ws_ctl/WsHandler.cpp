#include "WsHandler.h"

void WsHandler::onData(WebSocket *connection, const char *data) {
    if (0 == strcmp("die", data)) {
        _server->terminate();
        return;
    }
    if (0 == strcmp("close", data)) {
        std::cout << "Closing..\n";
        connection->close();
        std::cout << "Closed.\n";
        return;
    }

    const int value = std::stoi(data) + 1;
    if (value > _currentValue) {
        setValue(value);
        for (auto c : _connections) {
            c->send(std::to_string(_currentValue));
        }
    }
}

void WsHandler::onConnect(WebSocket *connection) {
    _connections.insert(connection);
    connection->send(std::to_string(_currentValue));
    std::cout << "Connected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress())
              << "\nCredentials: " << *(connection->credentials()) << "\n";
}

void WsHandler::onDisconnect(WebSocket *connection) {
    _connections.erase(connection);
    std::cout << "Disconnected: " << connection->getRequestUri()
              << " : " << formatAddress(connection->getRemoteAddress()) << "\n";
}
