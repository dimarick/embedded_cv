#ifndef HW_CTL_BROADCASTINGSERVER_H
#define HW_CTL_BROADCASTINGSERVER_H

#include <mutex>
#include <set>
#include <atomic>
#include <string>
#include <unordered_map>
#include <functional>
#include <thread>

namespace mini_server {
    class BroadcastingServer {
    public:
        typedef std::function<void(int socket, const std::string &)> MessageHandler;
        typedef std::function<void(int socket)> CloseHandler;
        static const size_t BUFFER_SIZE = 256;
        enum MessageTypeEnum : unsigned int {
            TYPE_MAT = 0,
            TYPE_TELEMETRY = 1,
            TYPE_ACK = 2,
            TYPE_CONTROL = 3,
        };
    private:
        struct MessageHeader {
            unsigned int magick;
            MessageTypeEnum type;
            unsigned int size;
            unsigned long ttl;
        };

        int socket = -1;
        std::mutex acceptedSocketsMutex;
        std::set<int> acceptedSockets;
        std::unordered_map<int, std::thread> threads;
        std::atomic<bool> running;
        MessageHandler onMessage;
        CloseHandler onClose;
    public:
        void setOnMessage(BroadcastingServer::MessageHandler onMessage);
        void setOnClose(BroadcastingServer::CloseHandler onClose);

        void setSocket(int _socket) {
            this->socket = _socket;
        }
        void run();
        void stop() {
            running = false;
        }

        void broadcast(const std::string &message);
        void broadcast(const void *buffer, size_t bufferSize, unsigned long ttl = 0, MessageTypeEnum type = TYPE_MAT);

        static void interact(int socket, BroadcastingServer *server, int threadId);

        void sendFrame(int s, const std::vector<char> &frame);

        std::vector<char>
        getTransportMessage(const void *buffer, size_t bufferSize, unsigned long ttl,
                            const MessageTypeEnum &type) const;
    };
}

#endif //HW_CTL_BROADCASTINGSERVER_H
