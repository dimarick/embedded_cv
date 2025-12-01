#ifndef HW_CTL_BROADCASTINGSERVER_H
#define HW_CTL_BROADCASTINGSERVER_H

#include <mutex>
#include <set>
#include <atomic>
#include <string>

namespace mini_server {
    class BroadcastingServer {
        struct MessageHeader {
            unsigned int magick;
            unsigned int size;
        };

        int socket = -1;
        std::mutex acceptedSocketsMutex;
        std::set<int> acceptedSockets;
        std::atomic<bool> running;
    public:
        void setSocket(int _socket) {
            this->socket = _socket;
        }
        void run();
        void stop() {
            running = false;
        }

        void broadcast(const std::string &message);
        void broadcast(const void *buffer, size_t bufferSize);
    };
}

#endif //HW_CTL_BROADCASTINGSERVER_H
