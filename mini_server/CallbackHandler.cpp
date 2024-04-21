#include "CallbackHandler.h"

using namespace mini_server;

void CallbackHandler::handle(int socket, const std::string &in, std::string &out) {
    handler(socket, in, out);
}
