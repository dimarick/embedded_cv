#include "CvCommandHandler.h"

void CvCommandHandler::handle(int socket, const std::string &in, std::string &out) {
    out = std::string("CVOK: ") + in;
    broadcastingServer.broadcast(std::string("TM: ") + out);
}
