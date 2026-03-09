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
    server.addWebSocketHandler("/cv_ctl", std::make_shared<SocketProxy>("/tmp/cv_ctl", 2));
    server.addWebSocketHandler("/cv_tm", std::make_shared<SocketProxy>("/tmp/cv_tm", 2));
    server.addWebSocketHandler("/cv_stream", std::make_shared<SocketProxy>("/tmp/cv_stream", 2));
    server.addWebSocketHandler("/cv_calib_tm", std::make_shared<SocketProxy>("/tmp/cv_calib_tm", 2));
    server.addWebSocketHandler("/cv_calib_stream", std::make_shared<SocketProxy>("/tmp/cv_calib_stream", 2));

    server.serve("web", 9090);

    signal(SIGINT, [](int signal) {
        server.terminate();
    });

    return 0;
}
