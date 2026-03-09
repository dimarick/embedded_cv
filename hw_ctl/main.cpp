#include <cpptrace/cpptrace.hpp>
#include <unistd.h>

#include <sstream>
#include <iomanip>
#include <chrono>

#include <CallbackHandler.h>
#include <CommandServer.h>
#include <IpcServer.h>
#include <SocketFactory.h>
#include <csignal>

using namespace mini_server;

std::string serializeTimePoint( const std::chrono::system_clock::time_point& time, const std::string& format)
{
    using namespace std::chrono;
    std::time_t tt = std::chrono::system_clock::to_time_t(time);
    std::tm tm = *std::gmtime(&tt); //GMT (UTC)
    std::stringstream ss;
    ss << std::put_time( &tm, format.c_str() );
    auto us = duration_cast<nanoseconds>(time.time_since_epoch());
    ss << std::setfill('0') << std::setw(6) << us.count() % (int)1e9;
    return ss.str();
}

static BroadcastingServer broadcastingServer;
static CommandServer commandServer;

static std::atomic<bool> running = true;

int main( int argc, char *argv[] ) {
    cpptrace::register_terminate_handler();

    broadcastingServer.setSocket(SocketFactory::createListeningSocket("/tmp/hw_tm", 10));
    commandServer.setSocket(SocketFactory::createListeningSocket("/tmp/hw_ctl", 1));

    auto handler = CallbackHandler(
        [](int socket, const std::string &in, std::string &out) {
            char prefix[] = "AT+";
            if (in.rfind(prefix) == 0) {
                out = "HWOK:";
                out += (in.substr(sizeof(prefix)-sizeof(""), in.size()));
            } else {
                out = "Invalid command";
            }
        }
    );

    commandServer.setHandler(handler);

    signal(SIGINT, [](int signal) {
        running = false;
        broadcastingServer.stop();
        commandServer.stop();
    });

    std::thread commandServerThread = std::thread([]() {
        commandServer.run();
    });

    std::thread broadcastingServerThread = std::thread([]() {
        broadcastingServer.run();
    });

    while(running) {
        sleep(2);
        auto message = serializeTimePoint(std::chrono::high_resolution_clock::now(), "%Z %Y-%m-%d %H:%M:%S.");
        broadcastingServer.broadcast(message);
    }

    broadcastingServerThread.join();
    commandServerThread.join();
}
