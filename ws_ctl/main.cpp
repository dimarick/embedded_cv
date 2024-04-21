#include <seasocks/PrintfLogger.h>
#include <seasocks/Server.h>
#include <cpptrace/cpptrace.hpp>
#include "SocketProxy.h"

int main(int argc, const char* argv[]) {
    cpptrace::register_terminate_handler();

    auto logger = std::make_shared<PrintfLogger>(Logger::Level::Debug);

    Server server(logger);

    server.addWebSocketHandler("/hw_ctl", std::make_unique<SocketProxy>(server, "hw_ctl"));
    server.addWebSocketHandler("/hw_tm", std::make_unique<SocketProxy>(server, "hw_tm"));
    server.addWebSocketHandler("/cv_ctl", std::make_unique<SocketProxy>(server, "cv_ctl"));
    server.addWebSocketHandler("/cv_tm", std::make_unique<SocketProxy>(server, "cv_tm"));
    server.serve("web", 9090);
    return 0;
}
