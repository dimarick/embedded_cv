#include <cpptrace/cpptrace.hpp>
#include <unistd.h>

#include <sstream>
#include <iomanip>
#include <chrono>

#include <CallbackHandler.h>
#include <CommandServer.h>
#include <BroadcastingServer.h>
#include <SocketFactory.h>

using namespace mini_server;

std::string serializeTimePoint( const std::chrono::system_clock::time_point& time, const std::string& format)
{
    std::time_t tt = std::chrono::system_clock::to_time_t(time);
    std::tm tm = *std::gmtime(&tt); //GMT (UTC)
    std::stringstream ss;
    ss << std::put_time( &tm, format.c_str() );
    return ss.str();
}

int main( int argc, char *argv[] ) {
    cpptrace::register_terminate_handler();
    BroadcastingServer broadcastingServer(SocketFactory::createListeningSocket("hw_tm", 10));

    CallbackHandler handler = CallbackHandler(
        [](int socket, const std::string &in, std::string &out) {
            char prefix[] = "AT+";
            if (in.rfind(prefix) == 0) {
                out = "HWOK:";
                out += (in.substr(sizeof(prefix), in.size()));
            } else {
                out = "Invalid command";
            }
        }
    );

    CommandServer commandServer(SocketFactory::createListeningSocket("hw_ctl", 1), handler);

    std::thread commandServerThread = std::thread([&commandServer]() {
        commandServer.run();
    });

    std::thread broadcastingServerThread = std::thread([&broadcastingServer]() {
        broadcastingServer.run();
    });

    while(true) {
        sleep(2);
        auto message = serializeTimePoint(std::chrono::high_resolution_clock::now(), "%Z %Y-%m-%d %H:%M:%S.");
        broadcastingServer.broadcast(message);
    }
}
