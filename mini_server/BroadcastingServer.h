#ifndef HW_CTL_BROADCASTINGSERVER_H
#define HW_CTL_BROADCASTINGSERVER_H

#include <mutex>
#include <set>

namespace mini_server {
    class BroadcastingServer {
        int socket;
        std::mutex acceptedSocketsMutex;
        std::set<int> acceptedSockets;
    public:
        explicit BroadcastingServer(int socket) : socket(socket) {}

        void run();

        void broadcast(const std::string &message);
    };
}

#endif //HW_CTL_BROADCASTINGSERVER_H
