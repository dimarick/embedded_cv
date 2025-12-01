#ifndef WS_CTL_SOCKETPROXY_H
#define WS_CTL_SOCKETPROXY_H

#include "ConnectionHandler.h"

#include <seasocks/WebSocket.h>

#include <iostream>
#include <string>
#include <utility>
#include <unistd.h>

using namespace seasocks;

class SocketProxy : public WebSocket::Handler {
private:
    const std::string socketName;
    std::unordered_map<WebSocket *, ConnectionHandler *> connectionHandlers;
public:
    explicit SocketProxy(std::string socketName) : socketName(std::move(socketName)) {}
    void onConnect(WebSocket* connection) override;
    void onData(WebSocket* connection, const uint8_t *data, size_t) override;
    void onDisconnect(WebSocket* connection) override;
};

#endif //WS_CTL_SOCKETPROXY_H
