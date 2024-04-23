#include <seasocks/PrintfLogger.h>
#include <seasocks/Server.h>
#include <cpptrace/cpptrace.hpp>
#include "SocketProxy.h"

int main(int argc, const char* argv[]) {
    cpptrace::register_terminate_handler();

    auto logger = std::make_shared<PrintfLogger>(Logger::Level::Debug);

    Server server(logger);

    auto nameHwCtl = std::string("/tmp/hw_ctl");
    auto nameHwTm = std::string("/tmp/hw_tm");
    auto nameCvCtl = std::string("/tmp/cv_ctl");
    auto nameCvTm = std::string("/tmp/cv_tm");

    auto handlerHwCtl = std::make_shared<SocketProxy>(server, nameHwCtl);
    auto handlerHwTm = std::make_shared<SocketProxy>(server, nameHwTm);
    auto handlerCvCtl = std::make_shared<SocketProxy>(server, nameCvCtl);
    auto handlerCvTm = std::make_shared<SocketProxy>(server, nameCvTm);

    server.addWebSocketHandler("/hw_ctl", handlerHwCtl);
    server.addWebSocketHandler("/hw_tm", handlerHwTm);
    server.addWebSocketHandler("/cv_ctl", handlerCvCtl);
    server.addWebSocketHandler("/cv_tm", handlerCvTm);

    server.serve("web", 9090);
    return 0;
}
