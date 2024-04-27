#ifndef HW_CTL_COMMANDSERVER_H
#define HW_CTL_COMMANDSERVER_H

#include <thread>
#include <unordered_map>
#include <atomic>
#include <unordered_set>
#include <mutex>
#include "HandlerInterface.h"

namespace mini_server {
    static const size_t BUFFER_SIZE = 256;
    class CommandServer {
        int socket = -1;
        HandlerInterface *handler = nullptr;
        std::atomic<bool> running;
        std::unordered_map<int, std::thread> threads;
        std::mutex deadThreadsMutex;
        std::unordered_set<int> deadThreads;

        static void interact(int socket, CommandServer *server, int threadId);
    public:
        void setSocket(int _socket) {
            socket = _socket;
        }
        void setHandler(HandlerInterface &_handler) {
            handler = &_handler;
        }

        void run();
        void stop() {
            running = false;
        }
    };
}

#endif //HW_CTL_COMMANDSERVER_H
