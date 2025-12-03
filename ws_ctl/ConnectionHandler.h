#ifndef EMBEDDED_CV_CONNECTIONHANDLER_H
#define EMBEDDED_CV_CONNECTIONHANDLER_H

#include <seasocks/WebSocket.h>
#include <seasocks/Server.h>
#include <thread>
#include <atomic>
#include <cstring>

using namespace seasocks;

class ConnectionHandler {
    struct MessageHeader {
        unsigned int magick;
        unsigned int size;
        unsigned long ttl;
    };
    WebSocket *connection;
    int socketFd = -1;
    std::thread readingThread;
    std::atomic<bool> running = true;
    int sendingQueueDepth;
    std::counting_semaphore<10> sendingMutex;
public:
    explicit ConnectionHandler(WebSocket *connection, int socketFd, int sendingQueueDepth = -1) :
        connection(connection),
        socketFd(socketFd),
        sendingQueueDepth(sendingQueueDepth),
        sendingMutex(this->sendingQueueDepth != -1 ? this->sendingQueueDepth : 10) {}
    void start();
    void stop();
    void onData(const uint8_t *data, size_t dataSize);
};


#endif //EMBEDDED_CV_CONNECTIONHANDLER_H
