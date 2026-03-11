#include <sys/un.h>
#include <stdexcept>
#include <sys/stat.h>
#include <format>
#include <libgen.h>

#include "SocketFactory.h"

using namespace mini_server;

extern "C++" int SocketFactory::createServerSocket(const std::string &name, int maxConnections) {
    struct stat socketStat{};

    char fileName[name.size() + 1];
    name.copy(fileName, sizeof (fileName));

    auto dir = dirname(fileName);

    if (stat(dir, &socketStat) != 0) {
        if (errno == ENOENT) {
            mkdir(dir, 0755);
        }
        if (errno != ENOENT && S_ISDIR(socketStat.st_mode)) {
            throw std::runtime_error(std::format("{} should be a directory for {} {}", dir, name, strerror(errno)));
        }
    }

    if (stat(name.c_str(), &socketStat) != 0 && errno != ENOENT) {
        throw std::runtime_error(std::format("Stat error for file {}, {}", name, strerror(errno)));
    }

    if (errno != ENOENT && S_ISFIFO(socketStat.st_mode)) {
        throw std::runtime_error(std::format("Unlink error for file {}, {}. File should be fifo or can be deleted", name, strerror(errno)));
    }

    auto sock = socket(AF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {
        throw std::runtime_error(strerror(errno));
    }

    sockaddr_un addr = {AF_UNIX, ""};

    name.copy(addr.sun_path, sizeof(addr.sun_path), 0);

    unlink(name.c_str());

    if (bind(sock, (sockaddr *)&addr, sizeof(addr)) < 0) {
        throw std::runtime_error(strerror(errno));
    }

    if (::listen(sock, maxConnections) < 0) {
        throw std::runtime_error(strerror(errno));
    }

    return sock;
}

extern "C++" int SocketFactory::createClientSocket(const std::string &name) {
    auto sock = socket(AF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {
        throw std::runtime_error(strerror(errno));
    }

    sockaddr_un addr = {AF_UNIX, ""};

    name.copy(addr.sun_path, sizeof(addr.sun_path), 0);

    if (connect(sock, (sockaddr *)&addr, sizeof(addr)) < 0) {
        throw std::runtime_error(strerror(errno));
    }

    return sock;
}
