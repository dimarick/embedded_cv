#include <seasocks/PrintfLogger.h>
#include <seasocks/Server.h>
#include <cpptrace/cpptrace.hpp>
#include <csignal>
#include "SocketProxy.h"

static Server server(std::make_shared<PrintfLogger>(Logger::Level::Debug));

int main(int argc, const char* argv[]) {
    cpptrace::register_terminate_handler();

//    server.addWebSocketHandler("/hw_ctl", std::make_shared<SocketProxy>("/tmp/hw_ctl"));
//    server.addWebSocketHandler("/hw_tm", std::make_shared<SocketProxy>("/tmp/hw_tm"));
    server.addWebSocketHandler("/cv_ctl", std::make_shared<SocketProxy>("/tmp/cv_ctl"));
    server.addWebSocketHandler("/cv_tm", std::make_shared<SocketProxy>("/tmp/cv_tm"));
    server.addWebSocketHandler("/stream", std::make_shared<SocketProxy>("/tmp/stream", 2));

    server.serve("web", 9090);

    signal(SIGINT, [](int signal) {
        server.terminate();
    });

    return 0;
}
