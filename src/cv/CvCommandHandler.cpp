#include "CvCommandHandler.h"

void CvCommandHandler::handle(int socket, const std::string &in, std::string &out) {
//    std::ostringstream o;
//
//    std::string _in = in;
//
//    char *token = std::strtok(_in.data(), " \n");
//
//    if (token == std::string("GETCONFIG")) {
//        getConfig(o);
//        out = o.str();
//    } else if (token == std::string("SETCONFIG")) {
//        char *name = std::strtok(nullptr, " ");
//        char *value = std::strtok(nullptr, " ");
//        if (name == std::string("denoiseLevel")) {
//            const auto fValue = std::stof(value);
//        }
//
//        getConfig(o);
//        commandServer.broadcast(o.str());
//    } else {
//        out = "ERROR INVALID COMMAND";
//    }
}

void CvCommandHandler::getConfig(std::ostringstream &o) {
//    o << "CONFIG denoiseLevel " << imageProcessor->denoiseLevel << std::endl;
}
