#ifndef HW_CTL_CALLBACKHANDLER_H
#define HW_CTL_CALLBACKHANDLER_H

#include "HandlerInterface.h"
#include <functional>

typedef const std::function<void(int socket, const std::string &in, std::string &out)> HandlerFunction;

namespace mini_server {
    class CallbackHandler : public HandlerInterface {
        const HandlerFunction &handler;
    public:
        explicit CallbackHandler(const HandlerFunction &handler) : handler(handler) {}

        void handle(int socket, const std::string &in, std::string &out) final;
    };
}

#endif //HW_CTL_CALLBACKHANDLER_H
