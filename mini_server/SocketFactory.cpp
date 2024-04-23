#include "SocketFactory.h"

using namespace mini_server;

extern "C++" int SocketFactory::createListeningSocket(const std::string &name, int maxConnections) {
    auto sock = socket(AF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {
        throw std::runtime_error(strerror(errno));
    }

    sockaddr addr = {AF_UNIX, ""};

    name.copy(addr.sa_data, sizeof(addr.sa_data), 0);

    unlink(name.c_str());

    if (bind(sock, &addr, sizeof(addr)) < 0) {
        throw std::runtime_error(strerror(errno));
    }

    if (::listen(sock, maxConnections) < 0) {
        throw std::runtime_error(strerror(errno));
    }

    return sock;
}
