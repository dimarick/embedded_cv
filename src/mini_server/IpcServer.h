#ifndef HW_CTL_BROADCASTINGSERVER_H
#define HW_CTL_BROADCASTINGSERVER_H

#include <mutex>
#include <set>
#include <atomic>
#include <condition_variable>
#include <string>
#include <unordered_map>
#include <functional>
#include <thread>

namespace mini_server {
    class IpcServer {
    public:
        typedef std::function<void(int socket, const std::string &)> StringMessageHandler;
        typedef std::function<void(int socket, const void *buffer, size_t size)> BinaryMessageHandler;
        typedef std::function<void(int socket)> CloseHandler;
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
            unsigned long expire;
        };
        struct SendingTask {
            std::atomic<bool> ready = false;
            std::atomic<bool> done = true;
            std::vector<char> buffer;
        };

        int socket = -1;
        mutable std::mutex mutex;
        std::set<int> acceptedSockets;
        mutable std::mutex sendingMutex;
        mutable std::condition_variable sendingCV;
        std::unordered_map<int, SendingTask> sendingTasks;
        std::unordered_map<int, std::thread> sendingThreads;
        std::unordered_map<int, std::thread> threads;
        std::set<int> deadThreads;
        std::atomic<bool> running = false;
        StringMessageHandler onStringMessage;
        BinaryMessageHandler onBinaryMessage;
        CloseHandler onClose;
        static void interactThread(int socket, IpcServer *server, int threadId);
        void interact(int socket) const;
    public:
        void setOnMessage(StringMessageHandler onMessage);
        void setOnMessage(BinaryMessageHandler onMessage);
        void setOnClose(CloseHandler onClose);

        void setSocket(int _socket) {
            this->socket = _socket;
        }
        void serve();
        void runClient();

        static void sendingThread(int interactionSocket, IpcServer *server);

        void stop() {
            running = false;
        }

        void broadcast(const std::string &message);
        void broadcast(const void *buffer, size_t bufferSize, unsigned long ttl = 0, MessageTypeEnum type = TYPE_MAT);

        void send(int s, const std::string &message, unsigned long ttl = 0, MessageTypeEnum type = TYPE_MAT);
        void send(int s, const void *buffer, size_t bufferSize, unsigned long ttl = 0, MessageTypeEnum type = TYPE_MAT);

        std::vector<char>
        createFrame(const void *buffer, size_t bufferSize, unsigned long expire,
                    const IpcServer::MessageTypeEnum &type) const;

        void sendFrame(int s, const std::vector<char> &frame);

        unsigned long getExpire(unsigned long ttl, long frameCreatedAt = 0) const;

        size_t getClientsCount() const;

        bool isRunning() {
            return running;
        }
    };
}

#endif //HW_CTL_BROADCASTINGSERVER_H
