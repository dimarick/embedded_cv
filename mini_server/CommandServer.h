#ifndef HW_CTL_COMMANDSERVER_H
#define HW_CTL_COMMANDSERVER_H

#include <thread>
#include <map>
#include <atomic>
#include "HandlerInterface.h"

namespace mini_server {
    static const size_t BUFFER_SIZE = 256;
    class CommandServer {
        int socket = -1;
        HandlerInterface *handler = nullptr;
        std::atomic<bool> running;

        static void interact(int socket, CommandServer *server, int threadId, std::map<int, std::thread> *threads);
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
