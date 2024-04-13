#include <seasocks/PrintfLogger.h>
#include <seasocks/Server.h>
#include <cpptrace/cpptrace.hpp>
#include "WsHandler.h"

int main(int argc, const char* argv[]) {
    cpptrace::register_terminate_handler();

    auto logger = std::make_shared<PrintfLogger>(Logger::Level::Debug);

    Server server(logger);

    auto handler = std::make_shared<WsHandler>(&server);
    server.addWebSocketHandler("/ws", handler);
    server.serve("web", 9090);
    return 0;
}
