// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

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
};

#endif //WS_CTL_SOCKETPROXY_H
