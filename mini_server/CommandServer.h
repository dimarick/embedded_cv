#ifndef HW_CTL_COMMANDSERVER_H
#define HW_CTL_COMMANDSERVER_H

#include <thread>
#include <map>
#include <atomic>
#include "HandlerInterface.h"

namespace mini_server {
    static const size_t BUFFER_SIZE = 256;
    class CommandServer {
        int socket;
        HandlerInterface &handler;
        static void interact(int socket, CommandServer *server, int threadId, std::map<int, std::thread> *threads);
        std::atomic<bool> running;
    public:
        CommandServer(int socket, HandlerInterface &handler) : socket(socket), handler(handler) {}

        void run();
        void stop();
    };
}

#endif //HW_CTL_COMMANDSERVER_H
