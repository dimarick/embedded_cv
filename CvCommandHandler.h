#ifndef EMBEDDED_CV_CVCOMMANDHANDLER_H
#define EMBEDDED_CV_CVCOMMANDHANDLER_H

#include <BroadcastingServer.h>
#include <HandlerInterface.h>

using namespace mini_server;

class CvCommandHandler : public HandlerInterface {
    BroadcastingServer &broadcastingServer;
public:
    explicit CvCommandHandler(BroadcastingServer &broadcastingServer) : broadcastingServer(broadcastingServer) {}

    void handle(int socket, const std::string &in, std::string &out) override;

};

#endif //EMBEDDED_CV_CVCOMMANDHANDLER_H
