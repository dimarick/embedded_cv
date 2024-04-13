#ifndef WS_CTL_WSHANDLER_H
#define WS_CTL_WSHANDLER_H

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

using namespace seasocks;

class WsHandler : public WebSocket::Handler {
public:
    explicit WsHandler(Server* server)
            : _server(server), _currentValue(0) {
        setValue(1);
    }

    void onConnect(WebSocket* connection) override;

    void onData(WebSocket* connection, const char* data) override;

    void onDisconnect(WebSocket* connection) override;

private:
    std::set<WebSocket*> _connections;
    Server* _server;
    int _currentValue;
    std::string _currentSetValue;

    void setValue(int value) {
        _currentValue = value;
    }
};

#endif //WS_CTL_WSHANDLER_H
