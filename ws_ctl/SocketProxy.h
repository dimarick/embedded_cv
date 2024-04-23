#ifndef WS_CTL_SOCKETPROXY_H
#define WS_CTL_SOCKETPROXY_H

#include <seasocks/PrintfLogger.h>
#include <seasocks/Server.h>
#include <seasocks/StringUtil.h>
#include <seasocks/WebSocket.h>
#include <seasocks/util/Json.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <atomic>
#include <unistd.h>
#include <netinet/in.h>

using namespace seasocks;

class SocketProxy : public WebSocket::Handler {
private:
    std::set<WebSocket*> _connections;
    Server &server;
    const std::string &socketName;
    int socketFd = -1;
    std::thread readingThread;
    std::atomic<bool> running = true;
public:
    explicit SocketProxy(Server &server, const std::string &socketName) : server(server), socketName(socketName) {}
    void onConnect(WebSocket* connection) override;
    void onData(WebSocket* connection, const char* data) override;
    void onDisconnect(WebSocket* connection) override;
};

#endif //WS_CTL_SOCKETPROXY_H
