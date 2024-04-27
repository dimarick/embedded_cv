#ifndef EMBEDDED_CV_CONNECTIONHANDLER_H
#define EMBEDDED_CV_CONNECTIONHANDLER_H

#include <seasocks/WebSocket.h>
#include <seasocks/Server.h>
#include <thread>
#include <atomic>
#include <cstring>

using namespace seasocks;

class ConnectionHandler {
    WebSocket *connection;
    int socketFd = -1;
    std::thread readingThread;
    std::atomic<bool> running = true;
public:
    ConnectionHandler(WebSocket *connection, int socketFd) : connection(connection), socketFd(socketFd) {}
    void start();
    void stop();
    void onData(const char* data);
};


#endif //EMBEDDED_CV_CONNECTIONHANDLER_H
