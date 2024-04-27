#ifndef WS_CTL_SOCKETPROXY_H
#define WS_CTL_SOCKETPROXY_H

#include "ConnectionHandler.h"

#include <seasocks/WebSocket.h>

#include <iostream>
#include <string>
#include <unistd.h>

using namespace seasocks;

class SocketProxy : public WebSocket::Handler {
private:
    const std::string socketName;
    std::unordered_map<WebSocket *, ConnectionHandler *> connectionHandlers;
public:
    explicit SocketProxy(const std::string socketName) : socketName(socketName) {}
    void onConnect(WebSocket* connection) override;
    void onData(WebSocket* connection, const char* data) override;
    void onDisconnect(WebSocket* connection) override;
};

#endif //WS_CTL_SOCKETPROXY_H
