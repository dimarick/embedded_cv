#ifndef HW_CTL_BROADCASTINGSERVER_H
#define HW_CTL_BROADCASTINGSERVER_H

#include <mutex>
#include <set>
#include <atomic>
#include <string>

namespace mini_server {
    class BroadcastingServer {
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
        void broadcast(const void *buffer, size_t bufferSize, unsigned long ttl = 0, MessageTypeEnum type = TYPE_MAT);
    };
}

#endif //HW_CTL_BROADCASTINGSERVER_H
