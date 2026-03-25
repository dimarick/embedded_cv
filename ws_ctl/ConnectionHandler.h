// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
// Copyright (c) 2026 Dmitrii Kosenok
//
// This file is part of EmbeddedCV.
//
// It is dual-licensed under the terms of the GNU General Public License v3
// and a commercial license. You can choose the license that fits your needs.
// For details, see the LICENSE file in the root of the repository.

#ifndef EMBEDDED_CV_CONNECTIONHANDLER_H
#define EMBEDDED_CV_CONNECTIONHANDLER_H

#include <seasocks/WebSocket.h>
#include <seasocks/Server.h>
#include <thread>
#include <atomic>
#include <cstring>

using namespace seasocks;

class ConnectionHandler {
public:
    enum MessageTypeEnum : unsigned int {
        TYPE_MAT = 0,
        TYPE_TELEMETRY = 1,
        TYPE_ACK = 2,
        TYPE_CONTROL = 3,
    };

    struct MessageHeader {
        unsigned int magick;
        MessageTypeEnum type;
        unsigned int size;
        unsigned long ttl;
    };
private:
    WebSocket *connection;
    int socketFd = -1;
    int outputWidth = 0;
    int outputHeight = 0;
    std::thread readingThread;
    std::atomic<bool> running = true;
    int sendingQueueDepth;
    std::counting_semaphore<10> sendingMutex;
public:
    explicit ConnectionHandler(WebSocket *connection, int socketFd, int outputWidth, int outputHeight, int sendingQueueDepth = -1) :
        connection(connection),
        socketFd(socketFd),
        outputWidth(outputWidth),
        outputHeight(outputHeight),
        sendingQueueDepth(sendingQueueDepth),
        sendingMutex(this->sendingQueueDepth != -1 ? this->sendingQueueDepth : 10) {}
    void start();
    void stop();
    void onData(const uint8_t *data, size_t dataSize);

    void selfStop();
};


#endif //EMBEDDED_CV_CONNECTIONHANDLER_H
