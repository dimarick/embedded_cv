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
    const int sendingQueueDepth;
    std::unordered_map<WebSocket *, ConnectionHandler *> connectionHandlers;
public:
    explicit SocketProxy(std::string socketName, int sendingQueueDepth = -1) : socketName(std::move(socketName)), sendingQueueDepth(sendingQueueDepth) {}
    void onConnect(WebSocket* connection) override;
    void onData(WebSocket* connection, const char*) override;
    void onData(WebSocket* connection, const uint8_t *data, size_t) override;
    void onDisconnect(WebSocket* connection) override;

    std::unordered_map<std::string, std::string> parseQuery(const WebSocket *connection) const;
};

#endif //WS_CTL_SOCKETPROXY_H
