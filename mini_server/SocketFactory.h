#ifndef EMBEDDED_CV_SOCKETFACTORY_H
#define EMBEDDED_CV_SOCKETFACTORY_H

#include <string>
#include <cstring>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/in.h>
#include <stdexcept>
#include "HandlerInterface.h"

namespace mini_server {
    class SocketFactory {
    public:
        static int createListeningSocket(const std::string &name, int maxConnections);
    };
}

#endif //EMBEDDED_CV_SOCKETFACTORY_H
