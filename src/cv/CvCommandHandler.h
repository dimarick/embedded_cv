#ifndef EMBEDDED_CV_CVCOMMANDHANDLER_H
#define EMBEDDED_CV_CVCOMMANDHANDLER_H

#include <BroadcastingServer.h>
#include <CommandServer.h>
#include <HandlerInterface.h>

using namespace mini_server;

class CvCommandHandler : public HandlerInterface {
    CommandServer &commandServer;
public:
    explicit CvCommandHandler(CommandServer &commandServer) : commandServer(commandServer) {}

    void handle(int socket, const std::string &in, std::string &out) override;

    void getConfig(std::ostringstream &o);
};

#endif //EMBEDDED_CV_CVCOMMANDHANDLER_H
