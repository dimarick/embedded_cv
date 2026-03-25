// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#include <seasocks/StringUtil.h>
#include <sys/un.h>
#include <format>
#include <netinet/tcp.h>
#include "SocketProxy.h"
#include "seasocks/Connection.h"

void SocketProxy::onData(WebSocket *connection, const uint8_t *data, size_t dataSize) {
    connectionHandlers.find(connection)->second->onData(data, dataSize);
}

void SocketProxy::onData(WebSocket *connection, const char *data) {
    connectionHandlers.find(connection)->second->onData((const uint8_t *)data, strlen(data));
}

void SocketProxy::onConnect(WebSocket *connection) {
    auto c = dynamic_cast<Connection *>(connection);
    int setTrue = 1;
    auto status = setsockopt(c->getFd(), SOL_TCP, TCP_NODELAY, &setTrue, sizeof(setTrue));

    if (status < 0) {
        perror("Unable to setsockopt(TCP_NODELAY)");
        connection->send(std::format("ERROR Unable to connect to socket: {}", strerror(errno)));
        connection->close();

        return;
    }


    sockaddr_un addr = {AF_UNIX};

    socketName.copy(addr.sun_path, sizeof(addr.sun_path), 0);
    auto socketFd = socket(AF_UNIX, SOCK_STREAM, 0);
    status = connect(socketFd, (const struct sockaddr *)&addr, sizeof(addr));

    if (status < 0) {
        perror("Unable to connect to socket");
        connection->send(std::format("ERROR Unable to connect to socket: {}", strerror(errno)));
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
